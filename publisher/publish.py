import asyncio, struct, time

NATS_URL = "nats://127.0.0.1:4222"
SUBJECT = "one"
FPS = 2
WIDTH, HEIGHT = 8, 8

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

# Frame 1: letter "J" — black on white
J = [
    [0,0,0,1,1,1,1,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,1,0,0,0],
    [1,0,0,0,1,0,0,0],
    [0,1,1,1,0,0,0,0],
]

# Frame 2: inverted "J" — white on black
J_INV = [
    [1,1,1,0,0,0,0,1],
    [1,1,1,1,0,1,1,1],
    [1,1,1,1,0,1,1,1],
    [1,1,1,1,0,1,1,1],
    [1,1,1,1,0,1,1,1],
    [1,1,1,1,0,1,1,1],
    [0,1,1,1,0,1,1,1],
    [1,0,0,0,1,1,1,1],
]

frame_a = make_p4(J)
frame_b = make_p4(J_INV)

async def main():
    import nats
    nc = await nats.connect(NATS_URL)
    print(f"Connected to {NATS_URL}, publishing to '{SUBJECT}' at {FPS} fps")

    interval = 1.0 / FPS
    toggle = False

    try:
        while True:
            payload = frame_b if toggle else frame_a
            await nc.publish(SUBJECT, payload)
            toggle = not toggle
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass
    finally:
        await nc.close()

if __name__ == "__main__":
    asyncio.run(main())
