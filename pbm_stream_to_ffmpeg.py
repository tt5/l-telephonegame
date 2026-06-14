#!/usr/bin/env python3
"""pbm_stream_to_ffmpeg.py

Reads PBM frames from a websocket (with embedded original 28x28),
classifies each frame, generates a new 28x28 image via CVAE,
and displays original + generated side by side at 56x28.

Message format:
    [PBM: variable bytes][orig_size: 4 bytes big-endian uint32][orig_data: orig_size bytes]
"""

import asyncio
import subprocess
import sys
from pathlib import Path

import numpy as np

from pbm_utils import parse_message, pbm_to_input

WS_URL = "ws://localhost:4195/get/ws"
SCRIPT_DIR = Path(__file__).parent
CLASSIFIER_PATH = SCRIPT_DIR / "mnist_model.onnx"
GENERATOR_PATH = SCRIPT_DIR / "cvae_generator.onnx"
LATENT_DIM = 16


def composite_grid(cls_in: np.ndarray, listener_in: np.ndarray,
                    orig: np.ndarray, cls_out: np.ndarray, listener_out: np.ndarray) -> bytes:
    """Composite 5 images into a 2x3 grid (84x28) RGB24.

    Layout:
        empty         | classifier_in  | listener_in
        original      | classifier_out | listener_out

    Each cell is 28x28. Empty cell is black.
    """
    W, H = 28, 28
    grid_w = W * 3  # 84
    grid_h = H * 2  # 56
    out = bytearray(grid_w * grid_h * 3)

    def paste(img: np.ndarray, col: int, row: int):
        """Paste a 28x28 float32 image into the grid at (col, row)."""
        for r in range(H):
            for c in range(W):
                val = int(img[r, c] * 255)
                val = max(0, min(255, val))
                gx = col * W + c
                gy = row * H + r
                offset = (gy * grid_w + gx) * 3
                out[offset] = val
                out[offset + 1] = val
                out[offset + 2] = val

    # Row 0: empty | classifier_in | listener_in
    paste(cls_in, 1, 0)
    paste(listener_in, 2, 0)

    # Row 1: original | classifier_out | listener_out
    paste(orig, 0, 1)
    paste(cls_out, 1, 1)
    paste(listener_out, 2, 1)

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
            "-video_size", "84x56",
            "-r", "2",
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=840:560:flags=neighbor",
            output_file,
        ]
    else:
        cmd = [
            "ffplay",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", "84x56",
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

            # Need at least "P4\n" to find the header
            if len(buf) < 4:
                continue

            idx = buf.find(b"P4\n")
            if idx == -1:
                buf.clear()
                continue

            pixel_data, orig_data, width, height = parse_message(bytes(buf[idx:]))
            if pixel_data is None:
                break
            assert width is not None and height is not None

            # Parse extended message: PBM + orig + cls_in + cls_out
            # Each section: [size: 4 bytes BE uint32][data: size bytes]
            bytes_per_row = (width + 7) // 8
            pbm_data_bytes = bytes_per_row * height
            header_end = bytes(buf[idx:]).index(b"\n", 3) + 1
            pbm_total = header_end + pbm_data_bytes

            pos = idx + pbm_total  # current read position

            def read_section():
                nonlocal pos
                if len(buf) < pos + 4:
                    return None
                size = int.from_bytes(bytes(buf[pos : pos + 4]), "big")
                pos += 4
                if len(buf) < pos + size:
                    return None
                data = bytes(buf[pos : pos + size])
                pos += size
                return data

            orig_section = read_section()
            cls_in_section = read_section()
            cls_out_section = read_section()

            if orig_section is None or cls_in_section is None or cls_out_section is None:
                break

            msg_len = pos - idx
            orig_data = orig_section

            # Reconstruct 28x28 images from message sections
            if orig_data is not None and len(orig_data) == 784:
                orig_28x28 = np.frombuffer(orig_data, dtype=np.uint8).reshape(28, 28).astype(np.float32) / 255.0
            else:
                orig_28x28 = np.zeros((28, 28), dtype=np.float32)

            if cls_in_section is not None and len(cls_in_section) == 784:
                cls_in_28x28 = np.frombuffer(cls_in_section, dtype=np.uint8).reshape(28, 28).astype(np.float32) / 255.0
            else:
                cls_in_28x28 = np.zeros((28, 28), dtype=np.float32)

            if cls_out_section is not None and len(cls_out_section) == 784:
                cls_out_28x28 = np.frombuffer(cls_out_section, dtype=np.uint8).reshape(28, 28).astype(np.float32) / 255.0
            else:
                cls_out_28x28 = np.zeros((28, 28), dtype=np.float32)

            # Listener: classify the PBM and generate its own CVAE output
            cls_input = pbm_to_input(pixel_data, width=width, height=height)
            cls_outputs = cls_session.run([cls_output_name], {cls_input_name: cls_input})
            cls_probs = cls_outputs[0][0]
            exp_probs = np.exp(cls_probs - np.max(cls_probs))
            cls_probs = exp_probs / exp_probs.sum()
            predicted = int(np.argmax(cls_probs))
            confidence = cls_probs[predicted]

            # Listener input image: what the classifier sees (after blur + invert)
            listener_in_28x28 = cls_input[0, :, :, 0]

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
            listener_out_28x28 = gen_outputs[0][0, :, :, 0]

            # Composite 2x3 grid
            rgb = composite_grid(cls_in_28x28, listener_in_28x28,
                                 orig_28x28, cls_out_28x28, listener_out_28x28)
            proc.stdin.write(rgb)
            proc.stdin.flush()

            frame_count += 1
            print(f"Frame {frame_count:4d}  predicted={predicted}  conf={confidence:.2f}", file=sys.stderr)

            buf = buf[idx + msg_len :]

    proc.stdin.close()
    proc.wait()


if __name__ == "__main__":
    asyncio.run(main())
