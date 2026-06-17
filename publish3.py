#!/usr/bin/env python3
"""publish3.py

Stage 3 publisher. Generates digits 0-11 using cvae3 + mnist3,
publishes to NATS subject "one".

Usage:
    uv run publish3.py
"""

import asyncio, time, logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from pbm_utils import downscale_to_pbm, pbm_to_input2, prepare_for_mnist2

NATS_URL = "nats://127.0.0.1:4222"
SUBJECT = "one"
FPS = 2
LATENT_DIM = 16
GENERATOR_PATH = Path(__file__).parent / "cvae3_generator.onnx"
CLASSIFIER_PATH = Path(__file__).parent / "mnist3_model.onnx"

# Derive num_classes from model output shape (no hardcoded NUM_CLASSES)
_cls_tmp = ort.InferenceSession(str(CLASSIFIER_PATH))
NUM_CLASSES = _cls_tmp.get_outputs()[0].shape[1]
print("publisher num_classes: ", NUM_CLASSES)
del _cls_tmp

NATS_URL = "nats://127.0.0.1:4222"
SUBJECT = "one"
FPS = 2
LATENT_DIM = 16
GENERATOR_PATH = Path(__file__).parent / "cvae3_generator.onnx"
CLASSIFIER_PATH = Path(__file__).parent / "mnist3_model.onnx"

# Derive num_classes from model output shape (no hardcoded NUM_CLASSES)
_cls_tmp = ort.InferenceSession(str(CLASSIFIER_PATH))
NUM_CLASSES = _cls_tmp.get_outputs()[0].shape[1]  # e.g. 12 for stage 3
del _cls_tmp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("publisher3")


def build_message(image_28x28: np.ndarray, publisher_digit: int, predicted: int, confidence: float) -> bytes:
    """Build a message containing publisher_digit + predicted + confidence + PBM + orig.

    Format:
        [publisher_digit: 1 byte]
        [predicted: 1 byte]
        [confidence: 2 bytes BE uint16, scaled by 10000]
        [PBM: variable bytes]
        [orig_size: 4 bytes BE uint32][orig_data: orig_size bytes]
    """
    pbm = downscale_to_pbm(image_28x28, width=22, height=22)
    orig_bytes = (image_28x28 * 255).clip(0, 255).astype(np.uint8).tobytes()
    orig_size = len(orig_bytes).to_bytes(4, "big")
    conf_int = min(65535, max(0, int(confidence * 10000)))
    return bytes([publisher_digit]) + bytes([predicted]) + conf_int.to_bytes(2, "big") + pbm + orig_size + orig_bytes


async def main():
    import nats
    import onnxruntime as ort

    log.info(f"Loading generator: {GENERATOR_PATH}")
    gen_session = ort.InferenceSession(str(GENERATOR_PATH))
    gen_output_name = gen_session.get_outputs()[0].name

    cls_session = ort.InferenceSession(str(CLASSIFIER_PATH))
    cls_input_name = cls_session.get_inputs()[0].name
    cls_output_name = cls_session.get_outputs()[0].name

    nc = await nats.connect(NATS_URL)
    log.info(f"Connected to {NATS_URL}, publishing digits 0-11 to '{SUBJECT}' at {FPS} fps (min confidence 70%)")

    interval = 1.0 / FPS
    idx = 0

    try:
        while True:
            digit = idx % NUM_CLASSES

            # Generate until classifier confidence >= 70%
            predicted = digit
            confidence = 0.0
            image_28x28 = np.zeros((28, 28), dtype=np.float32)
            for attempt in range(50):
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

                # Classify using mnist3 model (12 classes, 28x28 binary input)
                cls_input = prepare_for_mnist2(image_28x28)
                cls_outputs = cls_session.run([cls_output_name], {cls_input_name: cls_input})[0][0]
                exp_probs = np.exp(cls_outputs - np.max(cls_outputs))
                probs = exp_probs / exp_probs.sum()
                predicted = int(np.argmax(probs))
                confidence = probs[predicted]

                # Accept if predicted matches digit and confidence >= 70%
                if predicted == digit and confidence >= 0.7:
                    break
            else:
                log.warning(f"  digit={digit}  failed to reach 70% confidence, using best")
                log.info(f"  digit={digit}  predicted={predicted}  conf={confidence:.2f}")

            payload = build_message(image_28x28, digit, predicted, confidence)

            await nc.publish(SUBJECT, payload)
            if idx % 100 == 0:
                log.info(f"Published {idx} frames, last: digit={digit} pred={predicted} conf={confidence:.2f} attempts={attempt+1}")
            idx += 1
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass
    finally:
        await nc.close()


if __name__ == "__main__":
    asyncio.run(main())
