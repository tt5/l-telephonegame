import asyncio, time, logging
from pathlib import Path

import numpy as np
from digits import make_p4
from pbm_utils import downscale_to_pbm

NATS_URL = "nats://127.0.0.1:4222"
SUBJECT = "one"
FPS = 1
LATENT_DIM = 16
GENERATOR_PATH = Path(__file__).parent / "cvae_generator.onnx"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("publisher")


def build_message(image_28x28: np.ndarray) -> bytes:
    """Build a message containing the 8x8 PBM + the original 28x28 image.

    Format:
        [PBM 8x8: 14 bytes][orig_size: 4 bytes big-endian uint32][orig_data: orig_size bytes]
    """
    pbm = downscale_to_pbm(image_28x28, width=20, height=20)
    orig_bytes = (image_28x28 * 255).clip(0, 255).astype(np.uint8).tobytes()
    orig_size = len(orig_bytes).to_bytes(4, "big")
    return pbm + orig_size + orig_bytes


async def main():
    import nats
    import onnxruntime as ort

    log.info(f"Loading generator: {GENERATOR_PATH}")
    gen_session = ort.InferenceSession(str(GENERATOR_PATH))
    gen_input_names = [inp.name for inp in gen_session.get_inputs()]
    gen_output_name = gen_session.get_outputs()[0].name

    nc = await nats.connect(NATS_URL)
    log.info(f"Connected to {NATS_URL}, publishing digits 0-9 to '{SUBJECT}' at {FPS} fps")

    interval = 1.0 / FPS
    idx = 0

    try:
        while True:
            digit = idx % 10

            noise = np.random.normal(size=(1, LATENT_DIM)).astype(np.float32)
            label_oh = np.zeros((1, 10), dtype=np.float32)
            label_oh[0, digit] = 1.0

            gen_inputs = {}
            for inp in gen_session.get_inputs():
                if "latent" in inp.name.lower():
                    gen_inputs[inp.name] = noise
                elif "label" in inp.name.lower():
                    gen_inputs[inp.name] = label_oh

            gen_outputs = gen_session.run([gen_output_name], gen_inputs)
            image_28x28 = gen_outputs[0][0, :, :, 0]

            payload = build_message(image_28x28)

            await nc.publish(SUBJECT, payload)
            log.info(f"Published digit {digit} ({len(payload)} bytes)")
            idx += 1
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass
    finally:
        await nc.close()


if __name__ == "__main__":
    asyncio.run(main())
