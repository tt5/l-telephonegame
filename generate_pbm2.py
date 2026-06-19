#!/usr/bin/env python3
"""generate_pbm2.py — Stage 2

Generate 22x22 binary PBM images for training stage 3.
Replicates the stage-2 pipeline logic without NATS/realtime display.

Stage 2: cvae2 + mnist2
Output: data/pbm2/

Usage:
    uv run generate_pbm2.py [--count N] [--output-dir data/pbm2]
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import onnxruntime as ort
from pathlib import Path

from pbm_utils import downscale_to_pbm, pbm_to_input2, downscale_batch, pbm_to_input2_batch
from gen_pbm_common import (
    PBM_HEADER_LEN, PBM_DATA_BYTES, PBM_TOTAL,
    QUALITY_THRESHOLD, CONF_THRESHOLD,
    count_by_confidence, classify_image, classify_batch,
    generate_balanced, encode_pbm_batch, progress_report,
)

# ─── Stage config ──────────────────────────────────────────────────
STAGE = 2
GEN_MODEL = "cvae2_generator.onnx"
CLS_MODEL = "mnist2_model.onnx"
DEFAULT_OUT_DIR = "data/pbm2"
IMAGES_PER_CLASS = 100

# Derive num_classes from generator model label input shape
_gen_tmp = ort.InferenceSession(str(Path(__file__).parent / GEN_MODEL))
NUM_CLASSES = _gen_tmp.get_inputs()[1].shape[1]  # label_input shape
del _gen_tmp
print(f"Stage {STAGE}: generating {NUM_CLASSES} classes (generator label_input shape)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10000, help="Target total number of images")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_total = args.count
    target_high = target_total * (NUM_CLASSES - 1) // NUM_CLASSES
    target_low = target_total - target_high

    existing_high, existing_low = count_by_confidence(out_dir)
    print(f"Stage {STAGE}: {GEN_MODEL} + {CLS_MODEL}, {NUM_CLASSES} classes")
    print(f"Existing in {out_dir}: {existing_high + existing_low} total "
          f"(high-conf >= {CONF_THRESHOLD}: {existing_high}, low-conf < {CONF_THRESHOLD}: {existing_low})")
    print(f"Target: {target_total} total (high: {target_high}, low: {target_low})")

    high_need = max(0, target_high - existing_high)
    low_need = max(0, target_low - existing_low)

    if high_need == 0 and low_need == 0:
        print("Already have enough images. Nothing to do.")
        return

    print(f"Need: high={high_need}, low={low_need}")

    # Load models
    script_dir = Path(__file__).parent
    gen_session = ort.InferenceSession(str(script_dir / GEN_MODEL))
    gen_output_name = gen_session.get_outputs()[0].name

    cls_session = ort.InferenceSession(str(script_dir / CLS_MODEL))
    cls_input_name = cls_session.get_inputs()[0].name
    cls_output_name = cls_session.get_outputs()[0].name

    high_saved = 0
    low_saved = 0
    generated = 0
    red_flags = 0
    save_counter = 0
    start_time = time.time()

    print(f"Generating balanced images...")

    with ThreadPoolExecutor(max_workers=4) as writer_pool:
        while high_saved < high_need or low_saved < low_need:
            still_need = max(high_need - high_saved, 0) + max(low_need - low_saved, 0)
            imgs_per_class = max(1, still_need // (NUM_CLASSES * 2))

            good_images, good_digits = generate_balanced(
                gen_session, gen_output_name, cls_session, cls_input_name, cls_output_name,
                NUM_CLASSES, QUALITY_THRESHOLD, imgs_per_class
            )
            generated += len(good_images)

            if len(good_images) == 0:
                continue

            binary_grids = downscale_batch(good_images, width=22, height=22)
            accepted_pbm, pixel_data_list = encode_pbm_batch(binary_grids)
            accepted_digits = good_digits

            accepted_input_tensors = pbm_to_input2_batch(pixel_data_list, width=22, height=22)

            if len(accepted_input_tensors) > 0:
                batch_preds, batch_confs = classify_batch(
                    cls_session, cls_input_name, cls_output_name,
                    accepted_input_tensors
                )

                for i in range(len(accepted_pbm)):
                    digit = int(accepted_digits[i])
                    pbm_bytes = accepted_pbm[i]
                    final_predicted = int(batch_preds[i])
                    final_confidence = float(batch_confs[i])

                    if final_predicted != digit:
                        red_flags += 1
                        continue

                    is_high = final_confidence >= CONF_THRESHOLD
                    save = False
                    if is_high and high_saved < high_need:
                        save = True
                        high_saved += 1
                    elif not is_high and low_saved < low_need:
                        save = True
                        low_saved += 1

                    if save:
                        conf_int = min(65535, max(0, int(final_confidence * 10000)))
                        filename = f"{final_predicted}_{conf_int:04d}_{save_counter:06d}.pbm"
                        save_counter += 1
                        writer_pool.submit((out_dir / filename).write_bytes, pbm_bytes)

                        if (high_saved + low_saved) % 10000 == 0:
                            total_need = high_need + low_need
                            total_saved = high_saved + low_saved
                            print(f"  Progress: {total_saved}/{total_need} saved, "
                                  f"high: {high_saved}/{high_need}, "
                                  f"low: {low_saved}/{low_need}, "
                                  f"still needed: {total_need - total_saved}")

    elapsed = time.time() - start_time
    final_high, final_low = count_by_confidence(out_dir)
    print(f"\nDone! Generated {generated} images in {elapsed:.1f}s ({generated/elapsed:.0f} img/s)")
    print(f"Red flags: {red_flags}")
    print(f"Saved: {high_saved} high, {low_saved} low")
    print(f"Total in {out_dir}: {final_high + final_low} (high: {final_high}, low: {final_low})")


if __name__ == "__main__":
    main()
