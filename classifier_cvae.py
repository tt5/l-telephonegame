#!/usr/bin/env python3
"""classifier_cvae.py

Subscribes to NATS subject "one", classifies each 8x8 PBM frame
using the MNIST ONNX classifier, then generates a new digit image
using the CVAE generator. The generated image is downscaled to 8x8
and published to NATS subject "two" along with the original 28x28 image.

Message format (input and output):
    [PBM 8x8: 14 bytes][orig_size: 4 bytes big-endian uint32][orig_data: orig_size bytes]
"""

import asyncio
import sys
from pathlib import Path

import numpy as np

from pbm_utils import PBM_HEADER, downscale_to_pbm, parse_message, pbm_to_input

NATS_URL = "nats://127.0.0.1:4222"
SUBJECT_IN = "one"
SUBJECT_OUT = "two"
CLASSIFIER_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "mnist_model.onnx"
GENERATOR_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "cvae_generator.onnx"

LATENT_DIM = 16
PBM_TOTAL = len(PBM_HEADER) + 8  # header + 8 data bytes


def build_message(pbm: bytes, orig_data: bytes, cls_in_img: bytes, cls_out_img: bytes) -> bytes:
    """Build output message: PBM + original image + classifier input + classifier output.

    Format:
        [PBM: variable bytes]
        [orig_size: 4 bytes BE uint32][orig_data: orig_size bytes]
        [cls_in_size: 4 bytes BE uint32][cls_in_data: cls_in_size bytes]
        [cls_out_size: 4 bytes BE uint32][cls_out_data: cls_out_size bytes]
    """
    parts = bytearray()
    parts += pbm
    parts += len(orig_data).to_bytes(4, "big")
    parts += orig_data
    parts += len(cls_in_img).to_bytes(4, "big")
    parts += cls_in_img
    parts += len(cls_out_img).to_bytes(4, "big")
    parts += cls_out_img
    return bytes(parts)


async def main():
    import nats
    import onnxruntime as ort

    print(f"Loading classifier: {CLASSIFIER_PATH}")
    cls_session = ort.InferenceSession(str(CLASSIFIER_PATH))
    cls_input_name = cls_session.get_inputs()[0].name
    cls_output_name = cls_session.get_outputs()[0].name

    print(f"Loading generator: {GENERATOR_PATH}")
    gen_session = ort.InferenceSession(str(GENERATOR_PATH))
    gen_output_name = gen_session.get_outputs()[0].name

    nc = await nats.connect(NATS_URL)
    print(f"Connected to {NATS_URL}")
    print(f"Subscribed to '{SUBJECT_IN}', publishing to '{SUBJECT_OUT}'")

    frame_count = 0

    async def on_msg(msg):
        nonlocal frame_count
        data = msg.data

        pixel_data, orig_data, width, height = parse_message(data)
        if pixel_data is None:
            return
        assert width is not None and height is not None

        # Classify
        input_tensor = pbm_to_input(pixel_data, width=width, height=height)
        classifier_in_img = ((1.0 - input_tensor[0, :, :, 0]) * 255).clip(0, 255).astype(np.uint8).tobytes()

        cls_outputs = cls_session.run([cls_output_name], {cls_input_name: input_tensor})
        cls_probs = cls_outputs[0][0]
        exp_probs = np.exp(cls_probs - np.max(cls_probs))
        cls_probs = exp_probs / exp_probs.sum()
        predicted = int(np.argmax(cls_probs))
        confidence = cls_probs[predicted]

        # Generate new image from CVAE
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
        classifier_out_img = (image_28x28 * 255).clip(0, 255).astype(np.uint8).tobytes()

        # Build output: new PBM + original image passed through unchanged
        pbm = downscale_to_pbm(image_28x28, width=width, height=height)
        # Pass through the original image from the publisher (telephone game: preserve the original)
        if orig_data is not None and len(orig_data) == 784:
            passed_orig = orig_data
        else:
            passed_orig = (image_28x28 * 255).clip(0, 255).astype(np.uint8).tobytes()
        payload = build_message(pbm, passed_orig, classifier_in_img, classifier_out_img)

        frame_count += 1
        prob_str = " ".join(f"{i}:{cls_probs[i]:.2f}" for i in range(10))
        print(f"Frame {frame_count:4d}  predicted={predicted}  conf={confidence:.2f}  [{prob_str}]")

        await nc.publish(SUBJECT_OUT, payload)

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
