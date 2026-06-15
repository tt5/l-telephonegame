#!/usr/bin/env python3
"""generate_pbm.py

Generate 22x22 binary PBM images for training.
Uses cvae2_generator.onnx + mnist2_model.onnx (11 classes).
Saves all images with label and confidence in filename.

Usage:
    uv run generate_pbm.py [--count N] [--output-dir data/pbm]
"""

import argparse
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path

from pbm_utils import downscale_to_pbm, prepare_for_mnist2

LATENT_DIM = 16
NUM_CLASSES = 11

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100000, help="Number of images to generate")
    parser.add_argument("--output-dir", type=str, default="data/pbm", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Count existing files
    existing = len(list(out_dir.glob("*.pbm")))
    print(f"Existing images in {out_dir}: {existing}")
    print(f"Generating {args.count} new images...")

    # Load models
    gen_session = ort.InferenceSession(str(Path(__file__).parent / "cvae2_generator.onnx"))
    gen_output_name = gen_session.get_outputs()[0].name
    gen_input_names = [inp.name for inp in gen_session.get_inputs()]

    cls_session = ort.InferenceSession(str(Path(__file__).parent / "mnist2_model.onnx"))
    cls_input_name = cls_session.get_inputs()[0].name
    cls_output_name = cls_session.get_outputs()[0].name

    saved = 0
    start_time = time.time()

    for i in range(args.count):
        digit = i % NUM_CLASSES

        # Generate image from CVAE
        noise = np.random.normal(size=(1, LATENT_DIM)).astype(np.float32)
        label_oh = np.zeros((1, NUM_CLASSES), dtype=np.float32)
        label_oh[0, digit] = 1.0

        gen_inputs = {}
        for inp in gen_session.get_inputs():
            if "latent" in inp.name.lower():
                gen_inputs[inp.name] = noise
            elif "label" in inp.name.lower():
                gen_inputs[inp.name] = label_oh

        gen_outputs = gen_session.run([gen_output_name], gen_inputs)
        image_28x28 = gen_outputs[0][0, :, :, 0]

        # Classify
        cls_input = prepare_for_mnist2(image_28x28)
        cls_outputs = cls_session.run([cls_output_name], {cls_input_name: cls_input})[0][0]
        exp_probs = np.exp(cls_outputs - np.max(cls_outputs))
        probs = exp_probs / exp_probs.sum()
        predicted = int(np.argmax(probs))
        confidence = probs[predicted]

        # Save PBM with label and confidence in filename
        pbm_bytes = downscale_to_pbm(image_28x28, width=22, height=22)
        conf_int = min(65535, max(0, int(confidence * 10000)))
        filename = f"{predicted}_{conf_int:04d}_{int(time.time()*1000000):016d}.pbm"
        (out_dir / filename).write_bytes(pbm_bytes)
        saved += 1

        # Progress report
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  Generated {i+1}/{args.count} ({rate:.0f} img/s) - last: pred={predicted} conf={confidence:.3f}")

    elapsed = time.time() - start_time
    total = existing + saved
    print(f"\nDone! Saved {saved} images in {elapsed:.1f}s ({saved/elapsed:.0f} img/s)")
    print(f"Total images in {out_dir}: {total}")

if __name__ == "__main__":
    main()
