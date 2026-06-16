#!/usr/bin/env python3
"""generate_pbm.py — Stage 1

Generate 22x22 binary PBM images for training stage 2.
Replicates the stage-1 pipeline logic without NATS/realtime display.

For each image:
  1. Pick a digit (0-9, cycling)
  2. Generate from cvae1, classify with mnist1
  3. Quality check: retry infinitely until predicted==digit and confidence>=0.7
  4. Downscale 28x28 -> 22x22 PBM
  5. Upscale 22x22 -> 28x28, classify again with mnist1
  6. Red flag: if prediction != digit, don't save
  7. Save 22x22 PBM with prediction and confidence in filename

Saves images with a target ratio of 10/11 high-confidence (>=0.6) and
1/11 low-confidence (<0.6). Once enough of one type is collected, only
saves the other type. Logs progress throughout.

Usage:
    uv run generate_pbm.py [--count N] [--output-dir data/pbm]
"""

import argparse
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path

from pbm_utils import downscale_to_pbm, pbm_to_input2, downscale_batch, pbm_to_input2_batch

LATENT_DIM = 16
NUM_CLASSES = 10  # digits 0-9 (stage 1, no low_conf)
QUALITY_THRESHOLD = 0.7
CONF_THRESHOLD = 0.6
TARGET_RATIO_LOW = 1   # 1/11 low-conf
TARGET_RATIO_HIGH = 10  # 10/11 high_conf
BATCH_SIZE = 64  # generate+classify this many at once


def count_by_confidence(out_dir: Path):
    """Count existing high-conf and low-conf PBM files in the output directory."""
    high = 0
    low = 0
    for f in out_dir.glob("*.pbm"):
        parts = f.stem.split("_")
        if len(parts) >= 2:
            try:
                conf_int = int(parts[1])
                if conf_int >= int(CONF_THRESHOLD * 10000):
                    high += 1
                else:
                    low += 1
            except ValueError:
                pass
    return high, low


def classify_mnist1(cls_session, cls_input_name, cls_output_name, image_28x28):
    """Classify a 28x28 float32 image with mnist1 model. Returns (predicted, confidence)."""
    # mnist1 expects (1, 28, 28) float32 — no channel dimension
    input_tensor = image_28x28.reshape(1, 28, 28).astype(np.float32)
    cls_outputs = cls_session.run([cls_output_name], {cls_input_name: input_tensor})[0][0]
    exp_probs = np.exp(cls_outputs - np.max(cls_outputs))
    probs = exp_probs / exp_probs.sum()
    predicted = int(np.argmax(probs))
    confidence = float(probs[predicted])
    return predicted, confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10000, help="Target total number of images")
    parser.add_argument("--output-dir", type=str, default="data/pbm", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_total = args.count
    target_high = target_total * TARGET_RATIO_HIGH // (TARGET_RATIO_HIGH + TARGET_RATIO_LOW)
    target_low = target_total - target_high

    existing_high, existing_low = count_by_confidence(out_dir)
    print(f"Existing in {out_dir}: {existing_high + existing_low} total "
          f"(high-conf >= {CONF_THRESHOLD}: {existing_high}, low-conf < {CONF_THRESHOLD}: {existing_low})")
    print(f"Target: {target_total} total (high: {target_high}, low: {target_low})")
    print(f"Still needed: high={max(0, target_high - existing_high)}, "
          f"low={max(0, target_low - existing_low)}")

    if existing_high >= target_high and existing_low >= target_low:
        print("Already have enough images. Nothing to do.")
        return

    # Load models (stage 1)
    gen_session = ort.InferenceSession(str(Path(__file__).parent / "cvae_generator.onnx"))
    gen_output_name = gen_session.get_outputs()[0].name

    cls_session = ort.InferenceSession(str(Path(__file__).parent / "mnist_model.onnx"))
    cls_input_name = cls_session.get_inputs()[0].name
    cls_output_name = cls_session.get_outputs()[0].name

    high_need = max(0, target_high - existing_high)
    low_need = max(0, target_low - existing_low)
    high_saved = 0
    low_saved = 0
    generated = 0
    red_flags = 0
    save_counter = 0
    start_time = time.time()

    print(f"Generating images (batch_size={BATCH_SIZE})...")
    while high_saved < high_need or low_saved < low_need:
        # --- Batched generate + quality check ---
        # Generate BATCH_SIZE images from cvae1, classify all at once,
        # keep those where predicted==digit and confidence >= QUALITY_THRESHOLD
        still_need = (high_need - high_saved) + (low_need - low_saved)
        batch_target = min(BATCH_SIZE, still_need * 4)  # oversample to account for rejections
        batch_target = max(batch_target, 1)

        noise = np.random.normal(size=(batch_target, LATENT_DIM)).astype(np.float32)
        digits = np.array([(generated + i) % NUM_CLASSES for i in range(batch_target)])
        label_oh = np.zeros((batch_target, NUM_CLASSES), dtype=np.float32)
        label_oh[np.arange(batch_target), digits] = 1.0

        gen_inputs = {}
        for inp in gen_session.get_inputs():
            if "latent" in inp.name.lower():
                gen_inputs[inp.name] = noise
            elif "label" in inp.name.lower():
                gen_inputs[inp.name] = label_oh

        gen_outputs = gen_session.run([gen_output_name], gen_inputs)[0]  # (B, 28, 28, 1)
        images = gen_outputs[:, :, :, 0]  # (B, 28, 28)

        # Classify entire batch at once
        cls_input = images.reshape(batch_target, 28, 28).astype(np.float32)
        cls_logits = cls_session.run([cls_output_name], {cls_input_name: cls_input})[0]  # (B, 10)
        exp_probs = np.exp(cls_logits - cls_logits.max(axis=1, keepdims=True))
        probs = exp_probs / exp_probs.sum(axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        confs = probs[np.arange(batch_target), preds]

        # Filter: keep only images where predicted==digit and confidence >= threshold
        mask = (preds == digits) & (confs >= QUALITY_THRESHOLD)
        good_images = images[mask]       # (G, 28, 28)
        good_preds = preds[mask]
        good_confs = confs[mask]
        good_digits = digits[mask]
        generated += batch_target

        # --- Batch downscale + upscale all accepted images at once ---
        # Downscale: (G, 28, 28) float32 -> (G, 22, 22) uint8 binary
        binary_grids = downscale_batch(good_images, width=22, height=22)  # (G, 22, 22)

        # Encode binary grids back to PBM bytes for saving (vectorized with np.packbits)
        # Each row of 22 pixels packs into 3 bytes (MSB-first, left-aligned)
        # binary_grids: (G, 22, 22) uint8 -> packbits -> (G, 22, 3) uint8
        # Then strip 9-byte headers for pbm_to_input2_batch
        packed = np.packbits(binary_grids, axis=2, bitorder='big')  # (G, 22, 3)
        pixel_data_bytes = packed.tobytes()  # raw bytes, consecutive rows
        rows_bytes = 22 * 3  # 66 bytes per image
        accepted_pbm = []
        pixel_data_list = []
        header = b"P4\n22 22\n"
        for i in range(len(good_images)):
            offset = i * rows_bytes
            pbm = header + pixel_data_bytes[offset:offset + rows_bytes]
            accepted_pbm.append(pbm)
            pixel_data_list.append(pbm[9:])

        # Batch decode + upscale PBM pixel data for second classify
        accepted_input_tensors = pbm_to_input2_batch(pixel_data_list, width=22, height=22)  # (G, 28, 28)
        accepted_digits = good_digits

        # Batch classify all upscaled images at once
        if len(accepted_input_tensors) > 0:
            # accepted_input_tensors is already (G, 28, 28) from pbm_to_input2_batch
            batch_cls_logits = cls_session.run([cls_output_name], {cls_input_name: accepted_input_tensors})[0]  # (G, 10)
            batch_exp = np.exp(batch_cls_logits - batch_cls_logits.max(axis=1, keepdims=True))
            batch_probs = batch_exp / batch_exp.sum(axis=1, keepdims=True)
            batch_preds = np.argmax(batch_probs, axis=1)
            batch_confs = batch_probs[np.arange(len(batch_preds)), batch_preds]

            # Second pass: red flag check + save using batched results
            for i in range(len(accepted_pbm)):
                digit = accepted_digits[i]
                pbm_bytes = accepted_pbm[i]
                final_predicted = int(batch_preds[i])
                final_confidence = float(batch_confs[i])

                # Red flag check
                if final_predicted != digit:
                    red_flags += 1
                    if generated % 1000 < batch_target:
                        elapsed = time.time() - start_time
                        rate = generated / elapsed
                        total_saved = high_saved + low_saved
                        total_need = high_need + low_need
                        print(f"  Generated {generated} ({rate:.0f} img/s) | "
                              f"Saved {total_saved}/{total_need} "
                              f"(high: {high_saved}/{high_need}, low: {low_saved}/{low_need}) "
                              f"| red_flags={red_flags} "
                              f"| last: digit={digit} pred={final_predicted} conf={final_confidence:.3f} "
                              f"| [RED FLAG]")
                    continue

                # Save based on target ratio
                is_high = final_confidence >= CONF_THRESHOLD

                if is_high and high_saved < high_need:
                    save = True
                    high_saved += 1
                elif not is_high and low_saved < low_need:
                    save = True
                    low_saved += 1
                else:
                    save = False

                if save:
                    conf_int = min(65535, max(0, int(final_confidence * 10000)))
                    filename = f"{final_predicted}_{conf_int:04d}_{save_counter:06d}.pbm"
                    save_counter += 1
                    (out_dir / filename).write_bytes(pbm_bytes)

                # Progress report
                if (high_saved + low_saved) % 500 < 1 and (high_saved + low_saved) > 0:
                    elapsed = time.time() - start_time
                    rate = generated / elapsed
                    total_saved = high_saved + low_saved
                    total_need = high_need + low_need
                    print(f"  Generated {generated} ({rate:.0f} img/s) | "
                          f"Saved {total_saved}/{total_need} "
                          f"(high: {high_saved}/{high_need}, low: {low_saved}/{low_need}) "
                          f"| red_flags={red_flags} "
                          f"| last: digit={digit} pred={final_predicted} conf={final_confidence:.3f} "
                          f"| batch_yield={len(good_images)}/{batch_target} "
                          f"{'[SAVED]' if save else '[skipped]'}")

    elapsed = time.time() - start_time
    total_high = existing_high + high_saved
    total_low = existing_low + low_saved
    print(f"\nDone! Generated {generated} images in {elapsed:.1f}s ({generated/elapsed:.0f} img/s)")
    print(f"Red flags (wrong prediction after down/up-scale): {red_flags}")
    print(f"Saved this run: {high_saved} high-conf, {low_saved} low-conf")
    print(f"Total in {out_dir}: {total_high + total_low} "
          f"(high: {total_high}, low: {total_low})")

if __name__ == "__main__":
    main()
