#!/usr/bin/env python3
"""classifier_cvae3.py

Stage 3 classifier. Subscribes to NATS subject "one", classifies each
22x22 PBM frame using mnist3 (12 classes), generates a new image via cvae3,
and publishes to NATS subject "two" along with cls_in/cls_out images.

Usage:
    uv run classifier_cvae3.py [--max-images N]
"""

import asyncio
import sys
from pathlib import Path

import numpy as np

from pbm_utils import PBM_HEADER, downscale_to_pbm, parse_message, pbm_to_input2
from pbm_stream_to_ffmpeg3 import save_pbm

NATS_URL = "nats://127.0.0.1:4222"
SUBJECT_IN = "one"
SUBJECT_OUT = "two"
CLASSIFIER_PATH = Path(__file__).parent / "mnist3_model.onnx"
GENERATOR_PATH = Path(__file__).parent / "cvae3_generator.onnx"
LATENT_DIM = 16
NUM_CLASSES = 12  # digits 0-9 + low_conf + low_conf_2

# Parse --max-images from sys.argv at module level (before ONNX runtime init)
SAVED_COUNT = 0
_MAX_IMAGES = None
_i = 1
while _i < len(sys.argv):
    if sys.argv[_i] == "--max-images" and _i + 1 < len(sys.argv):
        _MAX_IMAGES = int(sys.argv[_i + 1])
        sys.argv = sys.argv[:_i] + sys.argv[_i + 2:]
    else:
        _i += 1

LATENT_DIM = 16

PBM_TOTAL = len(PBM_HEADER) + 8  # header + 8 data bytes


def build_message(pbm: bytes, orig_data: bytes, cls_in_img: bytes, cls_out_img: bytes,
                    publisher_digit: int, predicted: int, confidence: float) -> bytes:
    """Build output message: publisher_digit + predicted + confidence + PBM + orig + cls_in + cls_out.

    Format:
        [publisher_digit: 1 byte]
        [predicted: 1 byte]
        [confidence: 2 bytes BE uint16, scaled by 10000]
        [PBM: variable bytes]
        [orig_size: 4 bytes BE uint32][orig_data: orig_size bytes]
        [cls_in_size: 4 bytes BE uint32][cls_in_data: cls_in_size bytes]
        [cls_out_size: 4 bytes BE uint32][cls_out_data: cls_out_size bytes]
    """
    parts = bytearray()
    parts += bytes([publisher_digit])
    parts += bytes([predicted])
    conf_int = min(65535, max(0, int(confidence * 10000)))
    parts += conf_int.to_bytes(2, "big")
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
    if _MAX_IMAGES is not None:
        import pbm_stream_to_ffmpeg3 as _pbm_mod
        _pbm_mod.MAX_IMAGES = _MAX_IMAGES
        print(f"Max PBM images to save: {_MAX_IMAGES}")
    print(f"Subscribed to '{SUBJECT_IN}', publishing to '{SUBJECT_OUT}'")

    frame_count = 0

    async def on_msg(msg):
        nonlocal frame_count
        data = msg.data

        publisher_digit, cls_predicted, cls_confidence, pixel_data, orig_data, width, height = parse_message(data)
        if pixel_data is None or publisher_digit is None:
            return
        assert width is not None and height is not None

        # Classify using mnist3 model (12 classes, 28x28 binary input)
        input_tensor = pbm_to_input2(pixel_data, width=width, height=height)

        classifier_in_img = (input_tensor[0] * 255).clip(0, 255).astype(np.uint8).tobytes()

        cls_outputs = cls_session.run([cls_output_name], {cls_input_name: input_tensor})
        cls_logits = cls_outputs[0][0]
        exp_probs = np.exp(cls_logits - np.max(cls_logits))
        cls_probs = exp_probs / exp_probs.sum()
        predicted = int(np.argmax(cls_probs))
        confidence = cls_probs[predicted]

        gen_label = predicted  # can be 0-11

        # Save PBM if prediction matches publisher digit
        if predicted == publisher_digit:
            save_pbm(pixel_data, width, height, predicted, confidence)

        # Generate new image from CVAE (retry until confident)
        image_28x28 = np.zeros((28, 28), dtype=np.float32)
        candidate = image_28x28
        for _ in range(500):
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

            # Classify the generated image (mnist3 expects 3D input)
            cls_in = candidate.reshape(1, 28, 28).astype(np.float32)
            cls_out = cls_session.run([cls_output_name], {cls_input_name: cls_in})[0][0]
            exp_p = np.exp(cls_out - np.max(cls_out))
            gen_probs = exp_p / exp_p.sum()
            gen_predicted = int(np.argmax(gen_probs))
            gen_confidence = gen_probs[gen_predicted]

            if gen_predicted == predicted and gen_confidence >= 0.7:
                image_28x28 = candidate
                break
        else:
            image_28x28 = candidate  # use last attempt even if below threshold

        classifier_out_img = (image_28x28 * 255).clip(0, 255).astype(np.uint8).tobytes()

        # Build output: new PBM + original image passed through unchanged
        pbm = downscale_to_pbm(image_28x28, width=width, height=height)
        if orig_data is not None and len(orig_data) == 784:
            passed_orig = orig_data
        else:
            passed_orig = (image_28x28 * 255).clip(0, 255).astype(np.uint8).tobytes()
        payload = build_message(pbm, passed_orig, classifier_in_img, classifier_out_img, publisher_digit, predicted, confidence)

        frame_count += 1

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
