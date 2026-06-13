import asyncio, time, logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from digits import DIGITS, make_p4

NATS_URL = "nats://127.0.0.1:4222"
SUBJECT = "one"
FPS = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("publisher")

# Pre-build PBM frames for digits 0-9
frames = [make_p4(DIGITS[i]) for i in range(10)]

async def main():
    import nats
    nc = await nats.connect(NATS_URL)
    log.info(f"Connected to {NATS_URL}, publishing digits 0-9 to '{SUBJECT}' at {FPS} fps")

    interval = 1.0 / FPS
    idx = 0

    try:
        while True:
            digit = idx % 10
            payload = frames[digit]
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
