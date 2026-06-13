import asyncio, time, logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from digits import make_p4

NATS_URL = "nats://127.0.0.1:4222"
SUBJECT = "one"
FPS = 2
LATENT_DIM = 16
GENERATOR_PATH = Path(__file__).parent.parent / "cvae_generator.onnx"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("publisher")


def downscale_to_pbm(image_28x28: np.ndarray) -> bytes:
    """Downscale 28x28 float32 image to 8x8 binary PBM."""
    grid = np.zeros((8, 8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            r_start = r * 3 + min(r, 4)
            c_start = c * 3 + min(c, 4)
            r_end = min(r_start + 4, 28)
            c_end = min(c_start + 4, 28)
            grid[r, c] = image_28x28[r_start:r_end, c_start:c_end].mean()

    binary = (grid > 0.5).astype(np.uint8)

    buf = bytearray()
    buf += b"P4\n8 8\n"
    for row in binary:
        byte = 0
        for j in range(8):
            byte = (byte << 1) | (row[j] & 1)
        buf.append(byte)
    return bytes(buf)


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

            # Generate unique rendering from CVAE
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
            payload = downscale_to_pbm(image_28x28)

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
