import asyncio, struct, time, logging

NATS_URL = "nats://127.0.0.1:4222"
SUBJECT = "one"
FPS = 2
WIDTH, HEIGHT = 8, 8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("publisher")

# P4 PBM: each row packed into bytes, MSB first, 1-bit per pixel
# 0 = white, 1 = black

def make_p4(pixels):
    """Build a P4 PBM byte string from a 2D list of 0/1."""
    buf = bytearray()
    buf += f"P4\n{WIDTH} {HEIGHT}\n".encode("ascii")
    for row in pixels:
        padded = row + [0] * ((8 - len(row) % 8) % 8)
        for i in range(0, len(padded), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | (padded[i + j] & 1)
            buf.append(byte)
    return bytes(buf)

# 8x8 digit patterns 0-9 (1 = black, 0 = white)
DIGITS = {
    0: [
        [0,1,1,1,1,1,1,0],
        [1,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,1],
        [0,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0],
    ],
    1: [
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,1,0],
        [1,1,1,1,1,1,1,0],
        [1,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
    ],
    2: [
        [1,1,1,1,1,0,1,0],
        [1,0,0,0,1,0,1,0],
        [1,0,0,0,1,0,1,0],
        [1,0,0,0,1,0,1,0],
        [1,0,0,0,1,0,1,0],
        [1,0,0,0,1,0,1,0],
        [1,0,0,0,1,1,1,0],
        [0,0,0,0,0,0,0,0],
    ],
    3: [
        [0,1,1,1,1,1,0,0],
        [1,0,0,0,0,0,1,0],
        [1,0,0,0,0,0,1,0],
        [1,0,0,0,0,0,1,0],
        [1,0,0,0,0,0,1,0],
        [1,0,1,1,0,1,0,0],
        [0,1,0,0,1,0,0,0],
        [0,0,0,0,0,0,0,0],
    ],
    4: [
        [0,0,0,1,1,1,1,1],
        [0,0,1,0,0,1,0,0],
        [0,1,0,0,0,1,0,0],
        [1,0,0,0,0,1,0,0],
        [1,1,1,1,1,1,1,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,0],
    ],
    5: [
        [0,1,0,1,1,1,1,0],
        [1,0,0,1,0,0,1,0],
        [1,0,0,1,0,0,1,0],
        [1,0,0,1,0,0,1,0],
        [1,0,0,1,0,0,1,0],
        [1,0,0,1,0,0,1,0],
        [0,1,1,0,0,0,1,0],
        [0,0,0,0,0,0,0,0],
    ],
    6: [
        [0,1,1,1,1,1,0,0],
        [1,0,0,1,0,0,1,0],
        [1,0,0,1,0,0,1,0],
        [1,0,0,1,0,0,1,0],
        [1,0,0,1,0,0,1,0],
        [1,0,0,1,0,0,1,0],
        [0,1,1,0,0,1,0,0],
        [0,0,0,0,0,0,0,0],
    ],
    7: [
        [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,1,0],
        [1,1,1,0,0,0,1,0],
        [0,0,0,1,0,0,1,0],
        [0,0,0,0,1,0,1,0],
        [0,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,0],
    ],
    8: [
        [0,1,1,0,1,1,0,0],
        [1,0,0,1,0,0,1,0],
        [1,0,0,1,0,0,1,0],
        [1,0,0,1,0,0,1,0],
        [1,0,0,1,0,0,1,0],
        [1,0,0,1,0,0,1,0],
        [0,1,1,0,1,1,0,0],
        [0,0,0,0,0,0,0,0],
    ],
    9: [
        [0,1,1,0,0,1,0,0],
        [1,0,0,1,0,0,1,0],
        [1,0,0,1,0,0,1,0],
        [1,0,0,1,0,0,1,0],
        [1,0,0,1,0,0,1,0],
        [1,0,0,1,0,0,1,0],
        [0,1,1,1,1,1,0,0],
        [0,0,0,0,0,0,0,0],
    ],
}

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
