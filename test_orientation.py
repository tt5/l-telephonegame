#!/usr/bin/env python3
"""Send a single test frame to ffplay to check orientation.
Pattern: top-left pixel should be black, rest white.
If we see it in the correct corner, orientation is right.
"""

import asyncio
import subprocess
import sys

WS_URL = "ws://localhost:4195/get/ws"

# PBM header for 8x8
HEADER = b"P4\n8 8\n"

# Row 0: 10000000 = black pixel at left
# Rows 1-7: 00000000 = all white
FRAME = HEADER + bytes([0b10000000, 0, 0, 0, 0, 0, 0, 0])

# Alternative: column test
# Row 0: 10000000
# Row 1: 10000000
# ...
# Row 7: 10000000
# = vertical line on the left
FRAME_V = HEADER + bytes([0b10000000] * 8)

# Horizontal line at top
# Row 0: 11111111
# Rows 1-7: 00000000
FRAME_H = HEADER + bytes([0b11111111, 0, 0, 0, 0, 0, 0, 0])

async def main():
    import websockets

    cmd = [
        "ffplay",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", "8x8",
        "-framerate", "1",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    async with websockets.connect(WS_URL) as ws:
        # Send horizontal line (should appear at TOP of display)
        await ws.send(FRAME_H)
        proc.stdin.write(b"\x00" * 24 * 8)  # 8 rows of black... wait, we need RGB

    proc.stdin.close()
    proc.wait()

if __name__ == "__main__":
    asyncio.run(main())
