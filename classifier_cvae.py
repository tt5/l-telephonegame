#!/usr/bin/env python3
"""classifier_cvae.py

Subscribes to NATS subject "one", classifies each 8x8 PBM frame
using the MNIST ONNX classifier, then generates a new digit image
using the CVAE generator. The generated image is downscaled to 8x8
and published to NATS subject "two" along with the original 28x28 image.

Message format (input and output):
    [PBM 8x8: 14 bytes][orig_size: 4 bytes big-endian uint32][orig_data: orig_size bytes]
"""

import asyncio
import sys
from pathlib import Path

import numpy as np

NATS_URL = "nats://127.0.0.1:4222"
SUBJECT_IN = "one"
SUBJECT_OUT = "two"
CLASSIFIER_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "mnist_model.onnx"
GENERATOR_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "cvae_generator.onnx"

LATENT_DIM = 16
PBM_HEADER = b"P4\n8 8\n"
HEADER_LEN = len(PBM_HEADER)
PBM_DATA_BYTES = 8
PBM_TOTAL = HEADER_LEN + PBM_DATA_BYTES  # 14 bytes


def parse_message(data: bytes):
    """Parse a message into (pbm_pixel_data, original_bytes).
    Returns (None, None) if invalid."""
    idx = data.find(PBM_HEADER)
    if idx == -1:
        return None, None
    pixel_data = data[idx + HEADER_LEN : idx + HEADER_LEN + PBM_DATA_BYTES]
    if len(pixel_data) < PBM_DATA_BYTES:
        return None, None

    # Extract original image after the PBM
    orig_start = idx + PBM_TOTAL
    if len(data) < orig_start + 4:
        return pixel_data, None
    orig_size = int.from_bytes(data[orig_start : orig_start + 4], "big")
    orig_data = data[orig_start + 4 : orig_start + 4 + orig_size]
    if len(orig_data) < orig_size:
        return pixel_data, None

    return pixel_data, orig_data


def build_message(pbm: bytes, orig_data: bytes) -> bytes:
    """Build output message: PBM + original image size + original data."""
    orig_size = len(orig_data).to_bytes(4, "big")
    return pbm + orig_size + orig_data


def pbm_to_input(data: bytes) -> np.ndarray:
    """Convert 8x8 P4 PBM pixel data to (1, 28, 28, 1) float32 for the classifier."""
    grid = np.zeros((8, 8), dtype=np.float32)
    for row_idx, byte in enumerate(data):
        for col_idx in range(8):
            bit = (byte >> (7 - col_idx)) & 1
            grid[row_idx, col_idx] = bit

    out = np.zeros((28, 28), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            r_start = r * 3 + min(r, 4)
            c_start = c * 3 + min(c, 4)
            r_end = min(r_start + 4, 28)
            c_end = min(c_start + 4, 28)
            out[r_start:r_end, c_start:c_end] = grid[r, c]

    out = 1.0 - out
    return out.reshape(1, 28, 28, 1)


def downscale_to_pbm(image_28x28: np.ndarray) -> bytes:
    """Downscale 28x28 float32 image to 8x8 binary PBM."""
    grid = np.zeros((8, 8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            r_start = r * 3 + min(r, 4)
            c_start = c * 3 + min(c, 4)
            r_end = min(r_start + 4, 28)
            c_end = min(c_start + 4, 28)
            grid[r, c] = image_28x28[r_start:r_end, c_start:c_end].mean()

    binary = (grid > 0.5).astype(np.uint8)

    buf = bytearray()
    buf += PBM_HEADER
    for row in binary:
        byte = 0
        for j in range(8):
            byte = (byte << 1) | (row[j] & 1)
        buf.append(byte)
    return bytes(buf)


async def main():
    import nats
    import onnxruntime as ort

    print(f"Loading classifier: {CLASSIFIER_PATH}")
    cls_session = ort.InferenceSession(str(CLASSIFIER_PATH))
    cls_input_name = cls_session.get_inputs()[0].name
    cls_output_name = cls_session.get_outputs()[0].name

    print(f"Loading generator: {GENERATOR_PATH}")
    gen_session = ort.InferenceSession(str(GENERATOR_PATH))
    gen_output_name = gen_session.get_outputs()[0].name

    nc = await nats.connect(NATS_URL)
    print(f"Connected to {NATS_URL}")
    print(f"Subscribed to '{SUBJECT_IN}', publishing to '{SUBJECT_OUT}'")

    frame_count = 0

    async def on_msg(msg):
        nonlocal frame_count
        data = msg.data

        pixel_data, orig_data = parse_message(data)
        if pixel_data is None:
            return

        # Classify
        input_tensor = pbm_to_input(pixel_data)
        cls_outputs = cls_session.run([cls_output_name], {cls_input_name: input_tensor})
        cls_probs = cls_outputs[0][0]
        exp_probs = np.exp(cls_probs - np.max(cls_probs))
        cls_probs = exp_probs / exp_probs.sum()
        predicted = int(np.argmax(cls_probs))
        confidence = cls_probs[predicted]

        # Generate new image from CVAE
        noise = np.random.normal(size=(1, LATENT_DIM)).astype(np.float32)
        label_oh = np.zeros((1, 10), dtype=np.float32)
        label_oh[0, predicted] = 1.0

        gen_inputs = {}
        for inp in gen_session.get_inputs():
            if "latent" in inp.name.lower():
                gen_inputs[inp.name] = noise
            elif "label" in inp.name.lower():
                gen_inputs[inp.name] = label_oh

        gen_outputs = gen_session.run([gen_output_name], gen_inputs)
        image_28x28 = gen_outputs[0][0, :, :, 0]

        # Build output: new PBM + original image passed through
        pbm = downscale_to_pbm(image_28x28)

        # Use the NEW generated image as the "original" for the next stage
        # so the display can show the chain of transformations
        new_orig = (image_28x28 * 255).clip(0, 255).astype(np.uint8).tobytes()
        payload = build_message(pbm, new_orig)

        frame_count += 1
        prob_str = " ".join(f"{i}:{cls_probs[i]:.2f}" for i in range(10))
        print(f"Frame {frame_count:4d}  predicted={predicted}  conf={confidence:.2f}  [{prob_str}]")

        await nc.publish(SUBJECT_OUT, payload)

    sub = await nc.subscribe(SUBJECT_IN, cb=on_msg)

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await nc.close()


if __name__ == "__main__":
    asyncio.run(main())
