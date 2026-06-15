#!/usr/bin/env python3
"""pbm_stream_to_ffmpeg3.py

Stage 3 display. Reads PBM frames from a websocket (with embedded
original 28x28), classifies each frame using mnist3 (12 classes),
generates a new 28x28 image via cvae3, and displays original +
generated side by side at 56x28.

Collects metrics: FPS, red/yellow flags, retries, per-class distribution.

Usage:
    uv run pbm_stream_to_ffmpeg3.py [output.mp4]
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from pbm_utils import parse_message, pbm_to_input2

WS_URL = "ws://localhost:4195/get/ws"
SCRIPT_DIR = Path(__file__).parent
CLASSIFIER_PATH = SCRIPT_DIR / "mnist3_model.onnx"
GENERATOR_PATH = SCRIPT_DIR / "cvae3_generator.onnx"
LATENT_DIM = 16
NUM_CLASSES = 12  # digits 0-9 + low_conf + low_conf_2

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
    """Composite 5 images into a 2x3 grid (84x56) RGB24."""
    W, H = 28, 28
    grid_w = W * 3
    grid_h = H * 2
    out = bytearray(grid_w * grid_h * 3)

    if wrong_guess:
        bg_r, bg_g, bg_b = 255, 0, 0
    elif low_confidence:
        bg_r, bg_g, bg_b = 200, 200, 0
    else:
        bg_r, bg_g, bg_b = 0, 0, 0

    for r in range(H):
        for c in range(W):
            val = int(listener_in[r, c] * 255)
            val = max(0, min(255, val))
            gx = 2 * W + c
            gy = 0 * H + r
            offset = (gy * grid_w + gx) * 3
            if val < 128:
                out[offset] = bg_r
                out[offset + 1] = bg_g
                out[offset + 2] = bg_b
            else:
                out[offset] = val
                out[offset + 1] = val
                out[offset + 2] = val

    def paste(img, col, row):
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

    if cls_in_wrong:
        cls_bg_r, cls_bg_g, cls_bg_b = 255, 0, 0
    elif cls_in_low:
        cls_bg_r, cls_bg_g, cls_bg_b = 200, 200, 0
    else:
        cls_bg_r, cls_bg_g, cls_bg_b = 0, 0, 0

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

    paste(orig, 0, 1)
    paste(cls_out, 1, 1)
    paste(listener_out, 2, 1)
    return bytes(out)


def save_pbm(pixel_data, width, height, predicted_digit, confidence):
    """Save PBM image to data/pbm3 directory (up to MAX_IMAGES)."""
    global SAVED_COUNT, MAX_IMAGES
    if MAX_IMAGES is not None and SAVED_COUNT >= MAX_IMAGES:
        return
    import time as _time
    base = Path("data/pbm3")
    base.mkdir(parents=True, exist_ok=True)
    header = f"P4\n{width} {height}\n".encode("ascii")
    pbm_bytes = header + bytes(pixel_data)
    filename = f"{predicted_digit}_{confidence:.4f}_{int(_time.time()*1000)}.pbm"
    (base / filename).write_bytes(pbm_bytes)
    SAVED_COUNT += 1


class Metrics:
    def __init__(self):
        self.frame_count = 0
        self.red_flags = 0
        self.yellow_flags = 0
        self.cls_in_wrong = 0
        self.cls_in_low = 0
        self.retries = []
        self.predictions = [0] * NUM_CLASSES
        self.publisher_digits = [0] * NUM_CLASSES
        self.start_time = time.time()
        self.last_report_time = self.start_time

    def record(self, publisher_digit, predicted, confidence, cls_predicted,
               cls_confidence, wrong_guess, low_confidence_flag,
               cls_in_wrong_flag, cls_in_low_flag, retry_count):
        self.frame_count += 1
        if wrong_guess:
            self.red_flags += 1
        if low_confidence_flag:
            self.yellow_flags += 1
        if cls_in_wrong_flag:
            self.cls_in_wrong += 1
        if cls_in_low_flag:
            self.cls_in_low += 1
        self.retries.append(retry_count)
        if predicted < NUM_CLASSES:
            self.predictions[predicted] += 1
        if publisher_digit < NUM_CLASSES:
            self.publisher_digits[publisher_digit] += 1

        # Report every 5 seconds
        now = time.time()
        if now - self.last_report_time >= 5.0:
            self.report()
            self.last_report_time = now

    def report(self):
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        avg_retries = sum(self.retries) / len(self.retries) if self.retries else 0
        max_retries = max(self.retries) if self.retries else 0

        print(f"\n--- Metrics @ {elapsed:.0f}s ---", file=sys.stderr)
        print(f"  Frames: {self.frame_count}  FPS: {fps:.1f}", file=sys.stderr)
        print(f"  Red flags: {self.red_flags} ({100*self.red_flags/max(1,self.frame_count):.1f}%)", file=sys.stderr)
        print(f"  Yellow flags: {self.yellow_flags} ({100*self.yellow_flags/max(1,self.frame_count):.1f}%)", file=sys.stderr)
        print(f"  cls_in wrong: {self.cls_in_wrong} ({100*self.cls_in_wrong/max(1,self.frame_count):.1f}%)", file=sys.stderr)
        print(f"  cls_in low: {self.cls_in_low} ({100*self.cls_in_low/max(1,self.frame_count):.1f}%)", file=sys.stderr)
        print(f"  Avg retries: {avg_retries:.1f}  Max retries: {max_retries}", file=sys.stderr)

        # Per-class distribution
        print(f"  Predictions:", file=sys.stderr)
        for c in range(NUM_CLASSES):
            name = f"d{c}" if c < 10 else ("lc" if c == 10 else "lc2")
            pred_pct = 100 * self.predictions[c] / max(1, self.frame_count)
            pub_pct = 100 * self.publisher_digits[c] / max(1, self.frame_count)
            print(f"    {name}: pred={self.predictions[c]} ({pred_pct:.1f}%) pub={self.publisher_digits[c]} ({pub_pct:.1f}%)", file=sys.stderr)
        print(f"---\n", file=sys.stderr)

    def final_report(self):
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"FINAL METRICS", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        self.report()

        # Retry distribution
        if self.retries:
            buckets = {}
            for r in self.retries:
                bucket = r if r < 10 else (10 if r < 20 else (20 if r < 50 else 50))
                buckets[bucket] = buckets.get(bucket, 0) + 1
            print(f"  Retry distribution:", file=sys.stderr)
            for b in sorted(buckets.keys()):
                label = f"{b}" if b < 50 else "50+"
                print(f"    {label}: {buckets[b]} ({100*buckets[b]/len(self.retries):.1f}%)", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)


async def main():
    import onnxruntime as ort
    import websockets

    metrics = Metrics()

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
            "-r", "30",
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
        while True:
            data = await ws.recv()
            if isinstance(data, str):
                data = data.encode("latin-1")
            buf.extend(data)

            if len(buf) < 4:
                continue

            idx = buf.find(b"P4\n")
            if idx == -1:
                buf.clear()
                continue

            if idx == 0:
                buf.clear()
                continue

            msg_start = idx - 4
            if msg_start < 0:
                buf.clear()
                continue

            publisher_digit, cls_predicted, cls_confidence, pixel_data, orig_data, width, height = parse_message(bytes(buf[msg_start:]))
            if pixel_data is None:
                break
            assert width is not None and height is not None

            bytes_per_row = (width + 7) // 8
            pbm_data_bytes = bytes_per_row * height
            header_end = bytes(buf[idx:]).index(b"\n", 3) + 1
            pbm_total = header_end + pbm_data_bytes

            pos = idx + pbm_total

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

            msg_len = (pos - msg_start)
            orig_data = orig_section

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

            cls_input = pbm_to_input2(pixel_data, width=width, height=height)
            cls_outputs = cls_session.run([cls_output_name], {cls_input_name: cls_input})
            cls_logits = cls_outputs[0][0]
            exp_probs = np.exp(cls_logits - np.max(cls_logits))
            cls_probs = exp_probs / exp_probs.sum()
            predicted = int(np.argmax(cls_probs))
            confidence = cls_probs[predicted]

            listener_in_28x28 = cls_input[0]

            gen_label = predicted
            listener_out_28x28 = np.zeros((28, 28), dtype=np.float32)
            candidate = listener_out_28x28
            retry_count = 0
            for retry_count in range(50):
                noise = np.random.normal(size=(1, LATENT_DIM)).astype(np.float32)
                label_oh = np.zeros((1, NUM_CLASSES), dtype=np.float32)
                label_oh[0, gen_label] = 1.0

                gen_inputs = {}
                for inp in gen_session.get_inputs():
                    if "latent" in inp.name.lower():
                        gen_inputs[inp.name] = noise
                    elif "label" in inp.name.lower():
                        gen_inputs[inp.name] = label_oh

                gen_outputs = gen_session.run([gen_output_name], gen_inputs)
                candidate = gen_outputs[0][0, :, :, 0]

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

            wrong_guess = predicted != cls_predicted
            low_confidence_flag = (predicted == cls_predicted) and (confidence < 0.6)
            cls_in_wrong_flag = cls_predicted != publisher_digit
            cls_in_low_flag = (cls_predicted == publisher_digit) and (cls_confidence < 0.6)

            metrics.record(
                publisher_digit=publisher_digit,
                predicted=predicted,
                confidence=confidence,
                cls_predicted=cls_predicted,
                cls_confidence=cls_confidence,
                wrong_guess=wrong_guess,
                low_confidence_flag=low_confidence_flag,
                cls_in_wrong_flag=cls_in_wrong_flag,
                cls_in_low_flag=cls_in_low_flag,
                retry_count=retry_count,
            )

            if not wrong_guess:
                save_pbm(pixel_data, width, height, predicted, confidence)

            rgb = composite_grid(cls_in_28x28, listener_in_28x28,
                                 orig_28x28, cls_out_28x28, listener_out_28x28,
                                 wrong_guess=wrong_guess, low_confidence=low_confidence_flag,
                                 cls_in_wrong=cls_in_wrong_flag, cls_in_low=cls_in_low_flag)
            proc.stdin.write(rgb)
            proc.stdin.flush()

            buf = buf[idx + msg_len :]

    proc.stdin.close()
    proc.wait()
    metrics.final_report()


if __name__ == "__main__":
    asyncio.run(main())
