#!/usr/bin/env python3
"""pbm_stream_to_ffmpeg.py

Reads PBM frames from a websocket, splits them on frame boundaries,
converts to RGB, and pipes to ffmpeg for display or encoding.

Usage:
    uv run pbm_stream_to_ffmpeg.py [output.mp4]

If no output file is given, pipes raw RGB to stdout for ffplay.
"""

import asyncio
import subprocess
import sys

WS_URL = "ws://localhost:4195/get/ws"
WIDTH, HEIGHT = 8, 8

# P4 PBM header we expect: "P4\n8 8\n"
PBM_HEADER = b"P4\n8 8\n"
HEADER_LEN = len(PBM_HEADER)
# P4 row bytes = ceil(8/8) = 1 byte per row, 8 rows = 8 bytes
PBM_DATA_BYTES = 8
FRAME_TOTAL = HEADER_LEN + PBM_DATA_BYTES  # 14 bytes per frame


def pbm_to_rgb(data: bytes) -> bytes:
    """Convert P4 PBM pixel data (1-bit, 0=white 1=black) to RGB24.
    Each byte is one row of 8 pixels, MSB = leftmost.
    Output is row-major RGB24 for ffplay rawvideo."""
    # Build 8x8 pixel grid: grid[row][col]
    grid = []
    for byte in data:
        row = []
        for bit_pos in range(7, -1, -1):
            row.append((byte >> bit_pos) & 1)
        grid.append(row)

    # new cols = old rows (top to bottom)
    out = bytearray()
    for new_row in range(8):
        for new_col in range(8):
            bit = grid[new_row][new_col]
            if bit:
                out.extend(b"\x00\x00\x00")
            else:
                out.extend(b"\xff\xff\xff")
    return bytes(out)


async def main():
    import websockets

    output_file = sys.argv[1] if len(sys.argv) > 1 else None

    if output_file:
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-video_size", f"{WIDTH}x{HEIGHT}",
            "-r", "4",
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=160:160:flags=neighbor",
            output_file,
        ]
    else:
        cmd = [
            "ffplay",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{WIDTH}x{HEIGHT}",
            "-framerate", "2",
            "-",
        ]

    print(f"Starting: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )

    async with websockets.connect(WS_URL) as ws:
        buf = bytearray()
        while True:
            data = await ws.recv()
            if isinstance(data, str):
                data = data.encode("latin-1")
            buf.extend(data)

            while len(buf) >= FRAME_TOTAL:
                idx = buf.find(PBM_HEADER)
                if idx == -1:
                    buf.clear()
                    break
                if idx + FRAME_TOTAL > len(buf):
                    break

                frame_data = bytes(buf[idx + HEADER_LEN : idx + FRAME_TOTAL])
                rgb = pbm_to_rgb(frame_data)
                proc.stdin.write(rgb)
                proc.stdin.flush()

                buf = buf[idx + FRAME_TOTAL :]

    proc.stdin.close()
    proc.wait()


if __name__ == "__main__":
    asyncio.run(main())
