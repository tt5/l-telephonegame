#!/usr/bin/env python3
"""classifier.py

Subscribes to NATS subject "one", runs each 8x8 PBM frame through
an ONNX MNIST classifier, and publishes the predicted digit as a
new 8x8 PBM frame to NATS subject "two".

Usage:
    uv run classifier.py [model_path]

Default model path: mnist_model.onnx (in the same directory)
"""

import asyncio
import sys
from pathlib import Path

import numpy as np

from digits import DIGITS, make_p4

NATS_URL = "nats://127.0.0.1:4222"
SUBJECT_IN = "one"
SUBJECT_OUT = "two"
MODEL_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "mnist_model.onnx"


def pbm_to_input(data: bytes) -> np.ndarray:
    """Convert 8x8 P4 PBM pixel data to 28x28 float32 array for the model."""
    grid = np.zeros((8, 8), dtype=np.float32)
    for row_idx, byte in enumerate(data):
        for col_idx in range(8):
            bit = (byte >> (7 - col_idx)) & 1
            grid[row_idx, col_idx] = bit

    # Upscale 8x8 -> 28x28 using nearest-neighbor
    out = np.zeros((28, 28), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            r_start = r * 3 + min(r, 4)
            c_start = c * 3 + min(c, 4)
            r_end = min(r_start + 4, 28)
            c_end = min(c_start + 4, 28)
            out[r_start:r_end, c_start:c_end] = grid[r, c]

    # Invert: MNIST expects white-on-black (0=background), our PBM is 1=black
    out = 1.0 - out

    # Add batch dim: (1, 28, 28)
    return out.reshape(1, 28, 28)


async def main():
    import nats
    import onnxruntime as ort

    print(f"Loading ONNX model: {MODEL_PATH}")
    session = ort.InferenceSession(str(MODEL_PATH))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"  Input:  {input_name} {session.get_inputs()[0].shape}")
    print(f"  Output: {output_name} {session.get_outputs()[0].shape}")

    nc = await nats.connect(NATS_URL)
    print(f"Connected to {NATS_URL}")
    print(f"Subscribed to '{SUBJECT_IN}', publishing to '{SUBJECT_OUT}'")

    # Pre-build PBM frames for each digit
    digit_frames = {d: make_p4(DIGITS[d]) for d in range(10)}

    frame_count = 0

    async def on_msg(msg):
        nonlocal frame_count
        data = msg.data

        idx = data.find(b"P4\n8 8\n")
        if idx == -1:
            return
        header_len = 6
        pixel_data = data[idx + header_len : idx + header_len + 8]
        if len(pixel_data) < 8:
            return

        input_tensor = pbm_to_input(pixel_data)
        outputs = session.run([output_name], {input_name: input_tensor})
        probs = outputs[0][0]

        exp_probs = np.exp(probs - np.max(probs))
        probs = exp_probs / exp_probs.sum()

        predicted = int(np.argmax(probs))
        confidence = probs[predicted]

        frame_count += 1
        prob_str = " ".join(f"{i}:{probs[i]:.2f}" for i in range(10))
        print(f"Frame {frame_count:4d}  predicted={predicted}  conf={confidence:.2f}  [{prob_str}]")

        await nc.publish(SUBJECT_OUT, digit_frames[predicted])

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
