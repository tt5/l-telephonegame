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


def decode_pbm(data: bytes) -> np.ndarray:
    """Decode 8 raw P4 PBM bytes into an 8x8 float32 grid."""
    grid = np.zeros((8, 8), dtype=np.float32)
    for row_idx, byte in enumerate(data):
        for col_idx in range(8):
            bit = (byte >> (7 - col_idx)) & 1
            grid[row_idx, col_idx] = bit
    return grid


def upscale(grid: np.ndarray, dst_h: int, dst_w: int) -> np.ndarray:
    """Upscale a source grid to a larger destination by mapping each pixel to a block.

    Only supports upscaling (dst >= src). Extra pixels from non-even division
    are distributed to the first rows/columns to center the content.
    """
    src_h, src_w = grid.shape
    if dst_h < src_h or dst_w < src_w:
        raise ValueError(f"dst ({dst_h}x{dst_w}) must be >= src ({src_h}x{src_w})")
    out = np.zeros((dst_h, dst_w), dtype=np.float32)
    base_h, extra_h = divmod(dst_h, src_h)
    base_w, extra_w = divmod(dst_w, src_w)
    for r in range(src_h):
        for c in range(src_w):
            r_start = r * base_h + min(r, extra_h)
            c_start = c * base_w + min(c, extra_w)
            r_end = min(r_start + base_h + (1 if r < extra_h else 0), dst_h)
            c_end = min(c_start + base_w + (1 if c < extra_w else 0), dst_w)
            out[r_start:r_end, c_start:c_end] = grid[r, c]
    return out


def normalize_for_mnist(image: np.ndarray) -> np.ndarray:
    """Invert and reshape a 28x28 image to (1, 28, 28, 1) float32 for the classifier."""
    return (1.0 - image).reshape(1, 28, 28, 1)


def pbm_to_input(data: bytes) -> np.ndarray:
    """Convert 8x8 P4 PBM pixel data to (1, 28, 28, 1) float32 for the classifier."""
    return normalize_for_mnist(upscale(decode_pbm(data), 28, 28))
