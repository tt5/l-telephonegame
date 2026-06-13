#!/usr/bin/env python3
"""test_pattern.py - Send test patterns to check orientation.
Run from the nats directory: uv run test_pattern.py
Then view with: uv run pbm_stream_to_ffmpeg.py

Patterns sent:
  1. Top row black, rest white → should show thick line at top
  2. Left column black, rest white → should show thick line on left
  3. Diagonal from top-left to bottom-right
"""

import asyncio, time

WS_URL = "ws://localhost:4195/get/ws"
HEADER = b"P4\n8 8\n"

# Pattern 1: top row all black
TOP = HEADER + bytes([0b11111111, 0, 0, 0, 0, 0, 0, 0])

# Pattern 2: left column all black (MSB of each byte)
LEFT = HEADER + bytes([0b10000000] * 8)

# Pattern 3: main diagonal
DIAG = HEADER
for row in range(8):
    byte = 1 << (7 - row)  # bit position moves left to right
    DIAG += bytes([byte])

PATTERNS = [
    ("Top row black", TOP),
    ("Left col black", LEFT),
    ("Diagonal", DIAG),
]

async def main():
    import websockets
    async with websockets.connect(WS_URL) as ws:
        for name, frame in PATTERNS:
            print(f"Sending: {name}")
            await ws.send(frame)
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
