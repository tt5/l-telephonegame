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

from pbm_utils import parse_message, pbm_to_input2

WS_URL = "ws://localhost:4195/get/ws"
SCRIPT_DIR = Path(__file__).parent
CLASSIFIER_PATH = SCRIPT_DIR / "mnist2_model.onnx"
GENERATOR_PATH = SCRIPT_DIR / "cvae_generator.onnx"
LATENT_DIM = 16

# Parse --max-images from sys.argv at module level
SAVED_COUNT = 0
MAX_IMAGES = None
_i = 1
while _i < len(sys.argv):
    if sys.argv[_i] == "--max-images" and _i + 1 < len(sys.argv):
        MAX_IMAGES = int(sys.argv[_i + 1])
        sys.argv = sys.argv[:_i] + sys.argv[_i + 2:]
    else:
        _i += 1


def composite_grid(cls_in, listener_in, orig, cls_out, listener_out,
                    wrong_guess=False, low_confidence=False,
                    cls_in_wrong=False, cls_in_low=False):
    """Composite 5 images into a 2x3 grid (84x56) RGB24.

    Layout:
        empty         | classifier_in  | listener_in
        original      | classifier_out | listener_out

    wrong_guess: highlight listener_in background red
    low_confidence: highlight listener_in background yellow
    """
    W, H = 28, 28
    grid_w = W * 3  # 84
    grid_h = H * 2  # 56
    out = bytearray(grid_w * grid_h * 3)

    # Background color for listener_in cell (col=2, row=0)
    if wrong_guess:
        bg_r, bg_g, bg_b = 255, 0, 0  # red
    elif low_confidence:
        bg_r, bg_g, bg_b = 200, 200, 0  # yellow
    else:
        bg_r, bg_g, bg_b = 0, 0, 0  # black

    # Fill listener_in cell (col=2, row=0) with image, using bg color for dark pixels
    for r in range(H):
        for c in range(W):
            val = int(listener_in[r, c] * 255)
            val = max(0, min(255, val))
            gx = 2 * W + c
            gy = 0 * H + r
            offset = (gy * grid_w + gx) * 3
            if val < 128:
                # Dark pixel → use highlight background color
                out[offset] = bg_r
                out[offset + 1] = bg_g
                out[offset + 2] = bg_b
            else:
                # Light pixel → keep the digit
                out[offset] = val
                out[offset + 1] = val
                out[offset + 2] = val

    def paste(img, col, row):
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

    # Row 0: empty | cls_in (with highlight) | listener_in
    # Fill cls_in cell with highlight background for dark pixels
    if cls_in_wrong:
        cls_bg_r, cls_bg_g, cls_bg_b = 255, 0, 0  # red
    elif cls_in_low:
        cls_bg_r, cls_bg_g, cls_bg_b = 200, 200, 0  # yellow
    else:
        cls_bg_r, cls_bg_g, cls_bg_b = 0, 0, 0  # black

    for r in range(H):
        for c in range(W):
            val = int(cls_in[r, c] * 255)
            val = max(0, min(255, val))
            gx = 1 * W + c
            gy = 0 * H + r
            offset = (gy * grid_w + gx) * 3
            if val < 128:
                out[offset] = cls_bg_r
                out[offset + 1] = cls_bg_g
                out[offset + 2] = cls_bg_b
            else:
                out[offset] = val
                out[offset + 1] = val
                out[offset + 2] = val

    # Row 1: original | classifier_out | listener_out
    paste(orig, 0, 1)
    paste(cls_out, 1, 1)
    paste(listener_out, 2, 1)

    return bytes(out)


def save_pbm(pixel_data, width, height, predicted_digit, confidence):
    """Save PBM image to data/pbm directory (up to MAX_IMAGES)."""
    global SAVED_COUNT, MAX_IMAGES
    if MAX_IMAGES is not None and SAVED_COUNT >= MAX_IMAGES:
        return
    import time
    base = Path("data/pbm")
    base.mkdir(parents=True, exist_ok=True)
    header = f"P4\n{width} {height}\n".encode("ascii")
    pbm_bytes = header + bytes(pixel_data)
    filename = f"{predicted_digit}_{confidence:.4f}_{int(time.time()*1000)}.pbm"
    (base / filename).write_bytes(pbm_bytes)
    SAVED_COUNT += 1
    print(f"  Saved PBM #{SAVED_COUNT}: {filename}", file=sys.stderr)


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
            "-r", "10",
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

            if idx == 0:
                buf.clear()
                continue

            # Message format: [pub_digit: 1][predicted: 1][confidence: 2][P4\n...]
            # Find P4\n and go back 4 bytes to get the full message
            msg_start = idx - 4
            if msg_start < 0:
                buf.clear()
                continue

            publisher_digit, cls_predicted, cls_confidence, pixel_data, orig_data, width, height = parse_message(bytes(buf[msg_start:]))
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

            msg_len = (pos - msg_start)  # total from publisher_digit start to end of cls_out
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

            # Listener: classify the PBM using mnist2 model (11 classes)
            cls_input = pbm_to_input2(pixel_data, width=width, height=height)
            cls_outputs = cls_session.run([cls_output_name], {cls_input_name: cls_input})
            cls_logits = cls_outputs[0][0]
            exp_probs = np.exp(cls_logits - np.max(cls_logits))
            cls_probs = exp_probs / exp_probs.sum()
            predicted = int(np.argmax(cls_probs))
            confidence = cls_probs[predicted]

            # Listener input image: the 28x28 binary image
            listener_in_28x28 = cls_input[0]

            # Generate 28x28 from CVAE (retry until confident)
            # If model predicted low_conf (10), use second-highest digit for CVAE
            gen_label = predicted if predicted < 10 else int(np.argsort(cls_probs)[-2])
            listener_out_28x28 = np.zeros((28, 28), dtype=np.float32)
            candidate = listener_out_28x28
            for _ in range(50):
                noise = np.random.normal(size=(1, LATENT_DIM)).astype(np.float32)
                label_oh = np.zeros((1, 10), dtype=np.float32)
                label_oh[0, gen_label] = 1.0

                gen_inputs = {}
                for inp in gen_session.get_inputs():
                    if "latent" in inp.name.lower():
                        gen_inputs[inp.name] = noise
                    elif "label" in inp.name.lower():
                        gen_inputs[inp.name] = label_oh

                gen_outputs = gen_session.run([gen_output_name], gen_inputs)
                candidate = gen_outputs[0][0, :, :, 0]

                # Classify the generated image (mnist2 expects 3D input)
                cls_in = candidate.reshape(1, 28, 28).astype(np.float32)
                cls_out = cls_session.run([cls_output_name], {cls_input_name: cls_in})[0][0]
                exp_p = np.exp(cls_out - np.max(cls_out))
                gen_probs = exp_p / exp_p.sum()
                gen_predicted = int(np.argmax(gen_probs))
                gen_confidence = gen_probs[gen_predicted]

                if gen_predicted == predicted and gen_confidence >= 0.7:
                    listener_out_28x28 = candidate
                    break
            else:
                listener_out_28x28 = candidate

            # Listener highlight: based on listener's prediction vs classifier_cvae's prediction
            # (listener is compared to what classifier_cvae guessed, not the ground truth)
            wrong_guess = predicted != cls_predicted
            low_confidence = (predicted == cls_predicted) and (confidence < 0.6)

            # cls_in highlight: based on classifier_cvae's prediction vs publisher digit
            cls_in_wrong = cls_predicted != publisher_digit
            cls_in_low = (cls_predicted == publisher_digit) and (cls_confidence < 0.6)

            # Save PBM images that are not red-flagged
            if not wrong_guess:
                save_pbm(pixel_data, width, height, predicted, confidence)
            rgb = composite_grid(cls_in_28x28, listener_in_28x28,
                                 orig_28x28, cls_out_28x28, listener_out_28x28,
                                 wrong_guess=wrong_guess, low_confidence=low_confidence,
                                 cls_in_wrong=cls_in_wrong, cls_in_low=cls_in_low)
            proc.stdin.write(rgb)
            proc.stdin.flush()

            frame_count += 1
            print(f"Frame {frame_count:4d}  ground_truth={publisher_digit}  cls_cv={cls_predicted}({cls_confidence:.2f})  listener={predicted}({confidence:.2f})  cls_in={'OK' if cls_predicted == publisher_digit else 'WRONG'}  listener_in={'OK' if predicted == cls_predicted else 'WRONG'}", file=sys.stderr)

            buf = buf[idx + msg_len :]

    proc.stdin.close()
    proc.wait()


if __name__ == "__main__":
    asyncio.run(main())
