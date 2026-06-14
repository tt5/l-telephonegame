#!/usr/bin/env python3
"""test_pipeline.py — Test the full pipeline end-to-end.

Generates CVAE images, runs them through the PBM encode/decode pipeline,
and classifies them. This tells us the actual pipeline accuracy.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pbm_utils import downscale_to_pbm, pbm_to_input


def main():
    import onnxruntime as ort

    print("Loading models...")
    LATENT_DIM = 16

    gen_sess = ort.InferenceSession("cvae_generator.onnx")
    gen_input_names = [inp.name for inp in gen_sess.get_inputs()]
    gen_output_name = gen_sess.get_outputs()[0].name

    cls_sess = ort.InferenceSession("mnist_model.onnx")
    cls_input_name = cls_sess.get_inputs()[0].name
    cls_output_name = cls_sess.get_outputs()[0].name

    print(f"  Generator inputs:  {gen_input_names}")
    print(f"  Generator output:  {gen_output_name}")
    print(f"  Classifier input:  {cls_input_name} {cls_sess.get_inputs()[0].shape}")
    print(f"  Classifier output: {cls_output_name}")

    correct = 0
    total = 100  # 10 per digit

    print(f"\nGenerating {total} samples (10 per digit) and testing pipeline...")

    for digit in range(10):
        for i in range(10):
            # Generate 28x28 image from CVAE
            noise = np.random.normal(size=(1, LATENT_DIM)).astype(np.float32)
            label_oh = np.zeros((1, 10), dtype=np.float32)
            label_oh[0, digit] = 1.0

            gen_inputs = {}
            for inp in gen_sess.get_inputs():
                if "latent" in inp.name.lower():
                    gen_inputs[inp.name] = noise
                elif "label" in inp.name.lower():
                    gen_inputs[inp.name] = label_oh

            gen_outputs = gen_sess.run([gen_output_name], gen_inputs)
            image_28x28 = gen_outputs[0][0, :, :, 0]

            # === PIPELINE: CVAE image → PBM → classifier input ===

            pbm = downscale_to_pbm(image_28x28, width=24, height=24)
            input_tensor = pbm_to_input(pbm, width=24, height=24)

            # Classify
            cls_outputs = cls_sess.run([cls_output_name], {cls_input_name: input_tensor})
            cls_probs = cls_outputs[0][0]
            exp_probs = np.exp(cls_probs - np.max(cls_probs))
            cls_probs = exp_probs / exp_probs.sum()
            predicted = int(np.argmax(cls_probs))

            if predicted == digit:
                correct += 1

    accuracy = correct / total
    print(f"\n{'='*50}")
    print(f"Pipeline accuracy: {correct}/{total} = {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*50}")
    print(f"\nModel-only accuracy on MNIST test: ~97.74%")
    print(f"Pipeline accuracy:                {accuracy*100:.2f}%")
    if accuracy < 0.90:
        print(f"\nSignificant accuracy drop — problem is in the pipeline.")
    elif accuracy < 0.95:
        print(f"\nModerate accuracy drop — pipeline introduces some degradation.")
    else:
        print(f"\nSmall accuracy drop — pipeline is working well.")


if __name__ == "__main__":
    main()
