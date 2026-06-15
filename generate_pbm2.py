#!/usr/bin/env python3
"""generate_pbm2.py — Stage 2

Generate 22x22 binary PBM images for training stage 3.
Replicates the stage-2 pipeline logic without NATS/realtime display.

For each image:
  1. Pick a digit (0-10, cycling — 10 is low_conf)
  2. Generate from cvae2, classify with mnist2
  3. Quality check: retry infinitely until predicted==digit and confidence>=0.7
  4. Downscale 28x28 -> 22x22 PBM
  5. Upscale 22x22 -> 28x28, classify again with mnist2
  6. Red flag: if prediction != digit, don't save
  7. Save 22x22 PBM with prediction and confidence in filename

Saves images with a target ratio of 10/11 high-confidence (>=0.6) and
1/11 low-confidence (<0.6). Once enough of one type is collected, only
saves the other type. Logs progress throughout.

Usage:
    uv run generate_pbm2.py [--count N] [--output-dir data/pbm2]
"""

import argparse
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path

from pbm_utils import downscale_to_pbm, pbm_to_input2, prepare_for_mnist2

LATENT_DIM = 16
NUM_CLASSES = 11  # digits 0-9 + low_conf
QUALITY_THRESHOLD = 0.7
CONF_THRESHOLD = 0.6
TARGET_RATIO_LOW = 1   # 1/12 low-conf-2
TARGET_RATIO_HIGH = 11  # 11/12 high-conf


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


def classify_mnist2(cls_session, cls_input_name, cls_output_name, image_28x28):
    """Classify a 28x28 float32 image with mnist2 model. Returns (predicted, confidence)."""
    input_tensor = prepare_for_mnist2(image_28x28)
    cls_outputs = cls_session.run([cls_output_name], {cls_input_name: input_tensor})[0][0]
    exp_probs = np.exp(cls_outputs - np.max(cls_outputs))
    probs = exp_probs / exp_probs.sum()
    predicted = int(np.argmax(probs))
    confidence = float(probs[predicted])
    return predicted, confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50000, help="Target total number of images")
    parser.add_argument("--output-dir", type=str, default="data/pbm2", help="Output directory")
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

    # Load models (stage 2)
    script_dir = Path(__file__).parent
    gen_session = ort.InferenceSession(str(script_dir / "cvae2_generator.onnx"))
    gen_output_name = gen_session.get_outputs()[0].name

    cls_session = ort.InferenceSession(str(script_dir / "mnist2_model.onnx"))
    cls_input_name = cls_session.get_inputs()[0].name
    cls_output_name = cls_session.get_outputs()[0].name

    high_need = max(0, target_high - existing_high)
    low_need = max(0, target_low - existing_low)
    high_saved = 0
    low_saved = 0
    generated = 0
    red_flags = 0
    start_time = time.time()

    print(f"Generating images...")
    while high_saved < high_need or low_saved < low_need:
        digit = generated % NUM_CLASSES

        # --- Step 1: Generate + quality check (infinite retry) ---
        predicted = digit
        confidence = 0.0
        image_28x28 = np.zeros((28, 28), dtype=np.float32)
        attempts = 0
        while True:
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
            attempts += 1

            # Classify with mnist2 (11 classes: 0-9 + low_conf)
            predicted, confidence = classify_mnist2(cls_session, cls_input_name, cls_output_name, image_28x28)

            # Accept if predicted matches digit and confidence >= quality threshold
            if predicted == digit and confidence >= QUALITY_THRESHOLD:
                break
            # Otherwise retry infinitely

        # --- Step 2: Downscale 28x28 -> 22x22 PBM ---
        pbm_bytes = downscale_to_pbm(image_28x28, width=22, height=22)

        # --- Step 3: Upscale 22x22 -> 28x28 and classify again ---
        pbm_pixel_data = pbm_bytes[9:]  # strip P4\n22 22\n header (9 bytes)
        input_tensor = pbm_to_input2(pbm_pixel_data, width=22, height=22)
        cls_outputs = cls_session.run([cls_output_name], {cls_input_name: input_tensor})[0][0]
        exp_probs = np.exp(cls_outputs - np.max(cls_outputs))
        probs = exp_probs / exp_probs.sum()
        final_predicted = int(np.argmax(probs))
        final_confidence = float(probs[final_predicted])

        # --- Step 4: Red flag check ---
        if final_predicted != digit:
            red_flags += 1
            generated += 1
            if generated % 1000 == 0:
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

        # --- Step 5: Save based on target ratio ---
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
            filename = f"{final_predicted}_{conf_int:04d}_{int(time.time()*1000000):016d}.pbm"
            (out_dir / filename).write_bytes(pbm_bytes)

        generated += 1

        if generated % 1000 == 0:
            elapsed = time.time() - start_time
            rate = generated / elapsed
            total_saved = high_saved + low_saved
            total_need = high_need + low_need
            print(f"  Generated {generated} ({rate:.0f} img/s) | "
                  f"Saved {total_saved}/{total_need} "
                  f"(high: {high_saved}/{high_need}, low: {low_saved}/{low_need}) "
                  f"| red_flags={red_flags} "
                  f"| last: digit={digit} pred={final_predicted} conf={final_confidence:.3f} "
                  f"gen_attempts={attempts} "
                  f"{'[SAVED]' if save else '[skipped]'}")

    elapsed = time.time() - start_time
    final_high, final_low = count_by_confidence(out_dir)
    print(f"\nDone! Generated {generated} images in {elapsed:.1f}s ({generated/elapsed:.0f} img/s)")
    print(f"Red flags (wrong prediction after down/up-scale): {red_flags}")
    print(f"Saved this run: {high_saved} high-conf, {low_saved} low-conf")
    print(f"Total in {out_dir}: {final_high + final_low} "
          f"(high: {final_high}, low: {final_low})")


if __name__ == "__main__":
    main()
