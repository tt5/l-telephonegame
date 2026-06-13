#!/usr/bin/env python3
"""pbm_stream_to_ffmpeg.py

Reads PBM frames from a websocket, splits them on frame boundaries,
and pipes them to ffmpeg for encoding.

Usage:
    uv run pbm_stream_to_ffmpeg.py [output.mp4]

If no output file is given, pipes raw RGB to stdout for ffplay.
"""

import asyncio
import subprocess
import sys

WS_URL = "ws://localhost:4195/get/ws"
WIDTH, HEIGHT = 8, 8
FRAME_SIZE = WIDTH * HEIGHT * 3  # RGB24

# P4 PBM header we expect: "P4\n8 8\n"
PBM_HEADER = b"P4\n8 8\n"
HEADER_LEN = len(PBM_HEADER)
# P4 row bytes = ceil(8/8) = 1 byte per row, 8 rows = 8 bytes
PBM_DATA_BYTES = 8
FRAME_TOTAL = HEADER_LEN + PBM_DATA_BYTES  # 14 bytes per frame


def pbm_to_rgb(data: bytes) -> bytes:
    """Convert P4 PBM pixel data (1-bit, MSB first, 0=white 1=black) to RGB24."""
    out = bytearray()
    for byte in data:
        for bit_pos in range(7, -1, -1):
            bit = (byte >> bit_pos) & 1
            if bit:  # black
                out.extend(b"\x00\x00\x00")
            else:  # white
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
            "-r", "2",
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=160:160:flags=neighbor",
            output_file,
        ]
    else:
        cmd = [
            "ffplay",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{WIDTH}x{HEIGHT}",
            "-framerate", "2",
            "-",
        ]

    print(f"Starting ffmpeg: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    print(f"Connecting to {WS_URL}", file=sys.stderr)
    async with websockets.connect(WS_URL) as ws:
        buf = bytearray()
        frame_count = 0
        while True:
            data = await ws.recv()
            if isinstance(data, str):
                data = data.encode("latin-1")
            buf.extend(data)

            # Extract complete frames from buffer
            while len(buf) >= FRAME_TOTAL:
                # Find the next PBM header
                idx = buf.find(PBM_HEADER)
                if idx == -1:
                    buf.clear()
                    break
                if idx + FRAME_TOTAL > len(buf):
                    break  # incomplete frame, wait for more data

                frame_data = bytes(buf[idx + HEADER_LEN : idx + FRAME_TOTAL])
                rgb = pbm_to_rgb(frame_data)
                proc.stdin.write(rgb)
                proc.stdin.flush()

                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"Frames sent: {frame_count}", file=sys.stderr)

                buf = buf[idx + FRAME_TOTAL :]

    proc.stdin.close()
    proc.wait()
    print(f"Done. Total frames: {frame_count}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
