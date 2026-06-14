#!/usr/bin/env python3
"""pbm_stream_to_ffmpeg.py

Reads 8x8 PBM frames from a websocket (with embedded original 28x28),
classifies each frame, generates a new 28x28 image via CVAE,
and displays original + generated side by side at 56x28.

Message format:
    [PBM 8x8: 14 bytes][orig_size: 4 bytes big-endian uint32][orig_data: orig_size bytes]
"""

import asyncio
import subprocess
import sys
from pathlib import Path

import numpy as np

from pbm_utils import PBM_HEADER, PBM_TOTAL, parse_message, pbm_to_input

WS_URL = "ws://localhost:4195/get/ws"
SCRIPT_DIR = Path(__file__).parent
CLASSIFIER_PATH = SCRIPT_DIR / "mnist_model.onnx"
GENERATOR_PATH = SCRIPT_DIR / "cvae_generator.onnx"
LATENT_DIM = 16


def composite_side_by_side(orig_28x28: np.ndarray, gen_28x28: np.ndarray) -> bytes:
    """Composite two 28x28 grayscale images side by side into 56x28 RGB24."""
    out = bytearray()
    for r in range(28):
        for c in range(28):
            val = int(orig_28x28[r, c] * 255)
            val = max(0, min(255, val))
            out.extend(bytes([val, val, val]))
        for c in range(28):
            val = int(gen_28x28[r, c] * 255)
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
            "-video_size", "56x28",
            "-r", "2",
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=560:280:flags=neighbor",
            output_file,
        ]
    else:
        cmd = [
            "ffplay",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", "56x28",
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

            while len(buf) >= PBM_TOTAL:
                idx = buf.find(PBM_HEADER)
                if idx == -1:
                    buf.clear()
                    break

                pixel_data, orig_data = parse_message(bytes(buf[idx:]))
                if pixel_data is None:
                    break

                # Calculate total message length to advance buffer
                orig_start = idx + PBM_TOTAL
                if len(buf) < orig_start + 4:
                    break
                orig_size = int.from_bytes(bytes(buf[orig_start : orig_start + 4]), "big")
                msg_len = PBM_TOTAL + 4 + orig_size
                if len(buf) < idx + msg_len:
                    break

                # Reconstruct original 28x28
                if orig_data is not None and len(orig_data) == 784:
                    orig_28x28 = np.frombuffer(orig_data, dtype=np.uint8).reshape(28, 28).astype(np.float32) / 255.0
                else:
                    orig_28x28 = np.zeros((28, 28), dtype=np.float32)

                # Classify
                cls_input = pbm_to_input(pixel_data)
                cls_outputs = cls_session.run([cls_output_name], {cls_input_name: cls_input})
                cls_probs = cls_outputs[0][0]
                exp_probs = np.exp(cls_probs - np.max(cls_probs))
                cls_probs = exp_probs / exp_probs.sum()
                predicted = int(np.argmax(cls_probs))
                confidence = cls_probs[predicted]

                # Generate 28x28 from CVAE
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
                gen_28x28 = gen_outputs[0][0, :, :, 0]

                # Composite side by side
                rgb = composite_side_by_side(orig_28x28, gen_28x28)
                proc.stdin.write(rgb)
                proc.stdin.flush()

                frame_count += 1
                print(f"Frame {frame_count:4d}  predicted={predicted}  conf={confidence:.2f}", file=sys.stderr)

                buf = buf[idx + msg_len :]

    proc.stdin.close()
    proc.wait()


if __name__ == "__main__":
    asyncio.run(main())
