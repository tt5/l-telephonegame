#!/usr/bin/env python3
"""gen_pbm_common.py — Shared utilities for generate_pbm*.py

Common code for generating PBM training data across stages.
Used by generate_pbm.py (stage 1), generate_pbm2.py (stage 2), etc.

PBM format: P4 (binary), 22x22 pixels
  Header: "P4\n22 22\n" (9 bytes)
  Data: 22 rows × 3 bytes/row = 66 bytes
  Total: 75 bytes

Message format (for reference):
  [publisher_digit: 1 byte][predicted: 1 byte][confidence: 2 bytes BE uint16 / 10000]
  [PBM: variable bytes][orig_size: 4 bytes BE uint32][orig_data: orig_size bytes]
"""

import time
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ─── Constants ─────────────────────────────────────────────────────
PBM_HEADER = b"P4\n22 22\n"
PBM_HEADER_LEN = len(PBM_HEADER)  # 9 bytes
PBM_DATA_BYTES = 22 * 3  # 66 bytes (22 rows × 3 bytes/row)
PBM_TOTAL = PBM_HEADER_LEN + PBM_DATA_BYTES  # 75 bytes

QUALITY_THRESHOLD = 0.7
CONF_THRESHOLD = 0.6


def count_by_confidence(out_dir: Path, conf_threshold=CONF_THRESHOLD):
    """Count existing high-conf and low-conf PBM files in the output directory."""
    high = 0
    low = 0
    for f in out_dir.glob("*.pbm"):
        parts = f.stem.split("_")
        if len(parts) >= 2:
            try:
                conf_int = int(parts[1])
                if conf_int >= int(conf_threshold * 10000):
                    high += 1
                else:
                    low += 1
            except ValueError:
                pass
    return high, low


def classify_image(cls_session, cls_input_name, cls_output_name, image_28x28):
    """Classify a 28x28 float32 image. Returns (predicted, confidence).

    Uses prepare_for_mnist2 for binarization + reshape to (1, 28, 28).
    """
    from pbm_utils import prepare_for_mnist2
    input_tensor = prepare_for_mnist2(image_28x28)
    # Check if model expects 4D input (Conv2D) or 3D input (Flatten)
    input_shape = cls_session.get_inputs()[0].shape
    if len(input_shape) == 4:
        input_tensor = input_tensor[..., np.newaxis]  # (1, 28, 28) -> (1, 28, 28, 1)
    cls_outputs = cls_session.run([cls_output_name], {cls_input_name: input_tensor})[0][0]
    exp_probs = np.exp(cls_outputs - np.max(cls_outputs))
    probs = exp_probs / exp_probs.sum()
    predicted = int(np.argmax(probs))
    confidence = float(probs[predicted])
    return predicted, confidence


def classify_batch(cls_session, cls_input_name, cls_output_name, images):
    """Classify a batch of 28x28 float32 images. Returns (preds, confs).

    Args:
        images: (N, 28, 28) float32 array (no channel dimension)
    Returns:
        preds: (N,) int array of predicted classes
        confs: (N,) float array of confidence values
    """
    binary = (images > 0.5).astype(np.float32)
    # Check if model expects 4D input (Conv2D) or 3D input (Flatten)
    input_shape = cls_session.get_inputs()[0].shape
    if len(input_shape) == 4:
        binary = binary[..., np.newaxis]  # (N, 28, 28) -> (N, 28, 28, 1)
    cls_logits = cls_session.run([cls_output_name], {cls_input_name: binary})[0]
    exp_probs = np.exp(cls_logits - cls_logits.max(axis=1, keepdims=True))
    probs = exp_probs / exp_probs.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    confs = probs[np.arange(len(preds)), preds]
    return preds, confs


def generate_balanced(gen_session, gen_output_name, cls_session, cls_input_name, cls_output_name,
                      num_classes, quality_threshold, images_per_class):
    """Generate balanced images - exactly images_per_class per class."""
    all_images = [[] for _ in range(num_classes)]
    all_digits = [[] for _ in range(num_classes)]

    # Derive latent dim from generator model input
    latent_dim = None
    for inp in gen_session.get_inputs():
        if "latent" in inp.name.lower():
            latent_dim = inp.shape[1]
            break
    if latent_dim is None:
        raise ValueError("Could not find latent input in generator model")

    batch_size = num_classes * 10

    while min(len(imgs) for imgs in all_images) < images_per_class:
        # Find classes that still need more images
        needed = [(c, images_per_class - len(all_images[c])) for c in range(num_classes)]
        short_classes = [c for c, n in needed if n > 0]

        if not short_classes:
            break

        # Generate a batch cycling through all classes
        noise = np.random.normal(size=(batch_size, latent_dim)).astype(np.float32)
        digits = np.arange(batch_size) % num_classes
        label_oh = np.zeros((batch_size, num_classes), dtype=np.float32)
        label_oh[np.arange(batch_size), digits] = 1.0

        gen_inputs = {}
        for inp in gen_session.get_inputs():
            if "latent" in inp.name.lower():
                gen_inputs[inp.name] = noise
            elif "label" in inp.name.lower():
                gen_inputs[inp.name] = label_oh

        gen_outputs = gen_session.run([gen_output_name], gen_inputs)[0]
        images = gen_outputs[:, :, :, 0]

        preds, confs = classify_batch(cls_session, cls_input_name, cls_output_name, images)

        # Assign passing images to their class buckets
        for i in range(batch_size):
            c = int(digits[i])
            if preds[i] == c and confs[i] >= quality_threshold and len(all_images[c]) < images_per_class:
                all_images[c].append(images[i])
                all_digits[c].append(c)

        # Retry short classes individually
        for c in short_classes:
            #print(f"  [RETRY] Class {c} short ({len(all_images[c])}/{images_per_class}), retrying...")
            retry_count = 0
            while len(all_images[c]) < images_per_class:
                retry_count += 1
                noise = np.random.normal(size=(1, latent_dim)).astype(np.float32)
                label_oh = np.zeros((1, num_classes), dtype=np.float32)
                label_oh[0, c] = 1.0

                gen_inputs = {}
                for inp in gen_session.get_inputs():
                    if "latent" in inp.name.lower():
                        gen_inputs[inp.name] = noise
                    elif "label" in inp.name.lower():
                        gen_inputs[inp.name] = label_oh

                img = gen_session.run([gen_output_name], gen_inputs)[0][0, :, :, 0]
                pred, conf = classify_image(cls_session, cls_input_name, cls_output_name, img)

                if pred == c and conf >= quality_threshold:
                    all_images[c].append(img)
                    all_digits[c].append(c)
                    #print(f"  [RETRY] Class {c} passed after {retry_count} retries (conf={conf:.3f})")
                    break

    # Concatenate all classes
    good_images = np.array([img for imgs in all_images for img in imgs])
    good_digits = np.array([d for ds in all_digits for d in ds])
    return good_images, good_digits


def encode_pbm_batch(binary_grids):
    """Encode a batch of 22x22 binary grids to PBM bytes.

    Args:
        binary_grids: (N, 22, 22) uint8 array (0 or 1)
    Returns:
        pbm_list: list of bytes, each 75 bytes (9 header + 66 data)
        pixel_data_list: list of bytes, each 66 bytes (PBM data only, no header)
    """
    from pbm_utils import downscale_batch
    packed = np.packbits(binary_grids, axis=2, bitorder='big')  # (N, 22, 3)
    pixel_data_bytes = packed.tobytes()
    rows_bytes = PBM_DATA_BYTES  # 66 bytes per image

    pbm_list = []
    pixel_data_list = []
    for i in range(len(binary_grids)):
        offset = i * rows_bytes
        pbm = PBM_HEADER + pixel_data_bytes[offset:offset + rows_bytes]
        pbm_list.append(pbm)
        pixel_data_list.append(pbm[PBM_HEADER_LEN:])
    return pbm_list, pixel_data_list


def progress_report(generated, high_saved, low_saved, high_need, low_need, red_flags, start_time,
                    batch_target, digit=None, predicted=None, conf=None, prefix=""):
    """Print progress report."""
    elapsed = time.time() - start_time
    rate = generated / elapsed if elapsed > 0 else 0
    total_saved = high_saved + low_saved
    total_need = high_need + low_need
    msg = (f"  Generated {generated} ({rate:.0f} img/s) | "
           f"Saved {total_saved}/{total_need} "
           f"(high: {high_saved}/{high_need}, low: {low_saved}/{low_need}) "
           f"| red_flags={red_flags}")
    if digit is not None:
        msg += f" | last: digit={digit} pred={predicted} conf={conf:.3f}"
    if prefix:
        msg += f" | {prefix}"
    print(msg)
