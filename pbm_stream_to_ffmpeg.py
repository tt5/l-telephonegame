#!/usr/bin/env python3
"""pbm_stream_to_ffmpeg.py

Reads 8x8 PBM frames from a websocket, classifies each frame using
the MNIST ONNX classifier, then generates a new 28x28 grayscale image
using the CVAE generator. The result is displayed via ffplay at 28x28.

This is the "listener" in the telephone game — it sees an 8x8 digit,
guesses what it is, and draws its own version at 28x28.

Usage:
    uv run pbm_stream_to_ffmpeg.py [output.mp4]

Models are loaded from the same directory:
    - mnist_model.onnx (classifier)
    - cvae_generator.onnx (generator)
"""

import asyncio
import subprocess
import sys
from pathlib import Path

import numpy as np

WS_URL = "ws://localhost:4195/get/ws"
SCRIPT_DIR = Path(__file__).parent
CLASSIFIER_PATH = SCRIPT_DIR / "mnist_model.onnx"
GENERATOR_PATH = SCRIPT_DIR / "cvae_generator.onnx"
LATENT_DIM = 16

# P4 PBM header
PBM_HEADER = b"P4\n8 8\n"
HEADER_LEN = len(PBM_HEADER)
PBM_DATA_BYTES = 8
FRAME_TOTAL = HEADER_LEN + PBM_DATA_BYTES


def pbm_to_classifier_input(data: bytes) -> np.ndarray:
    """Convert 8x8 P4 PBM pixel data to (1, 28, 28, 1) float32 for the classifier."""
    grid = np.zeros((8, 8), dtype=np.float32)
    for row_idx, byte in enumerate(data):
        for col_idx in range(8):
            bit = (byte >> (7 - col_idx)) & 1
            grid[row_idx, col_idx] = bit

    # Upscale 8x8 -> 28x28 (nearest-neighbor)
    out = np.zeros((28, 28), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            r_start = r * 3 + min(r, 4)
            c_start = c * 3 + min(c, 4)
            r_end = min(r_start + 4, 28)
            c_end = min(c_start + 4, 28)
            out[r_start:r_end, c_start:c_end] = grid[r, c]

    # Invert: MNIST expects white-on-black
    out = 1.0 - out
    return out.reshape(1, 28, 28, 1)


def grayscale_to_rgb24(image_2d: np.ndarray) -> bytes:
    """Convert 28x28 float32 grayscale to RGB24 bytes for ffplay."""
    out = bytearray()
    for r in range(28):
        for c in range(28):
            val = int(image_2d[r, c] * 255)
            val = max(0, min(255, val))
            out.extend(bytes([val, val, val]))
    return bytes(out)


async def main():
    import onnxruntime as ort
    import websockets

    print(f"Loading classifier: {CLASSIFIER_PATH}")
    cls_session = ort.InferenceSession(str(CLASSIFIER_PATH))
    cls_input_name = cls_session.get_inputs()[0].name
    cls_output_name = cls_session.get_outputs()[0].name

    print(f"Loading generator: {GENERATOR_PATH}")
    gen_session = ort.InferenceSession(str(GENERATOR_PATH))
    gen_output_name = gen_session.get_outputs()[0].name

    output_file = sys.argv[1] if len(sys.argv) > 1 else None

    if output_file:
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-video_size", "28x28",
            "-r", "2",
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=280:280:flags=neighbor",
            output_file,
        ]
    else:
        cmd = [
            "ffplay",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", "28x28",
            "-framerate", "60",
            "-",
        ]

    print(f"Starting: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )

    print(f"Connecting to {WS_URL}")
    async with websockets.connect(WS_URL) as ws:
        buf = bytearray()
        frame_count = 0
        while True:
            data = await ws.recv()
            if isinstance(data, str):
                data = data.encode("latin-1")
            buf.extend(data)

            while len(buf) >= FRAME_TOTAL:
                idx = buf.find(PBM_HEADER)
                if idx == -1:
                    buf.clear()
                    break
                if idx + FRAME_TOTAL > len(buf):
                    break

                pixel_data = bytes(buf[idx + HEADER_LEN : idx + FRAME_TOTAL])

                # Classify the 8x8 image
                cls_input = pbm_to_classifier_input(pixel_data)
                cls_outputs = cls_session.run([cls_output_name], {cls_input_name: cls_input})
                cls_probs = cls_outputs[0][0]
                exp_probs = np.exp(cls_probs - np.max(cls_probs))
                cls_probs = exp_probs / exp_probs.sum()
                predicted = int(np.argmax(cls_probs))
                confidence = cls_probs[predicted]

                # Generate 28x28 image from CVAE using predicted label
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

                rgb = grayscale_to_rgb24(image_28x28)
                proc.stdin.write(rgb)
                proc.stdin.flush()

                frame_count += 1
                print(f"Frame {frame_count:4d}  predicted={predicted}  conf={confidence:.2f}", file=sys.stderr)

                buf = buf[idx + FRAME_TOTAL :]

    proc.stdin.close()
    proc.wait()


if __name__ == "__main__":
    asyncio.run(main())
