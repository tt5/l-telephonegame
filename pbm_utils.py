"""Shared PBM parsing and conversion utilities."""

import numpy as np

PBM_HEADER = b"P4\n8 8\n"
HEADER_LEN = len(PBM_HEADER)
PBM_DATA_BYTES = 8
PBM_TOTAL = HEADER_LEN + PBM_DATA_BYTES  # 14 bytes


def parse_message(data: bytes):
    """Parse a message into (pbm_pixel_data, original_bytes).
    Returns (None, None) if invalid."""
    idx = data.find(PBM_HEADER)
    if idx == -1:
        return None, None
    pixel_data = data[idx + HEADER_LEN : idx + HEADER_LEN + PBM_DATA_BYTES]
    if len(pixel_data) < PBM_DATA_BYTES:
        return None, None

    orig_start = idx + PBM_TOTAL
    if len(data) < orig_start + 4:
        return pixel_data, None
    orig_size = int.from_bytes(data[orig_start : orig_start + 4], "big")
    orig_data = data[orig_start + 4 : orig_start + 4 + orig_size]
    if len(orig_data) < orig_size:
        return pixel_data, None

    return pixel_data, orig_data


def pbm_to_input(data: bytes) -> np.ndarray:
    """Convert 8x8 P4 PBM pixel data to (1, 28, 28, 1) float32 for the classifier."""
    grid = np.zeros((8, 8), dtype=np.float32)
    for row_idx, byte in enumerate(data):
        for col_idx in range(8):
            bit = (byte >> (7 - col_idx)) & 1
            grid[row_idx, col_idx] = bit

    out = np.zeros((28, 28), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            r_start = r * 3 + min(r, 4)
            c_start = c * 3 + min(c, 4)
            r_end = min(r_start + 4, 28)
            c_end = min(c_start + 4, 28)
            out[r_start:r_end, c_start:c_end] = grid[r, c]

    out = 1.0 - out
    return out.reshape(1, 28, 28, 1)
