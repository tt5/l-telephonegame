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


def decode_pbm(data: bytes, width: int = 8, height: int = 8) -> np.ndarray:
    """Decode raw P4 PBM pixel data into a float32 grid of shape (height, width).

    Args:
        data: Raw PBM pixel bytes (without header). Must contain enough bytes
              for the given width and height (bytes per row = ceil(width / 8)).
        width: Image width in pixels. Default 8.
        height: Image height in pixels. Default 8.
    """
    bytes_per_row = (width + 7) // 8
    grid = np.zeros((height, width), dtype=np.float32)
    expected = bytes_per_row * height
    if len(data) < expected:
        raise ValueError(f"Need {expected} bytes for {width}x{height} PBM, got {len(data)}")
    for row_idx in range(height):
        row_start = row_idx * bytes_per_row
        for col_idx in range(width):
            byte_idx = row_start + col_idx // 8
            bit = (data[byte_idx] >> (7 - col_idx % 8)) & 1
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


def dither_binary(image: np.ndarray) -> np.ndarray:
    """Apply Floyd-Steinberg error-diffusion dithering to produce a binary image.

    Input values are clamped to [0, 1]. Output is strictly 0.0 or 1.0.
    Quantization error is diffused to neighboring pixels:
        7/16 right, 3/16 below-left, 5/16 below, 1/16 below-right.
    """
    img = image.copy().clip(0, 1)
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            old = img[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            img[y, x] = new
            err = old - new
            if x + 1 < w:
                img[y, x + 1] += err * 7 / 16
            if y + 1 < h:
                if x - 1 >= 0:
                    img[y + 1, x - 1] += err * 3 / 16
                img[y + 1, x] += err * 5 / 16
                if x + 1 < w:
                    img[y + 1, x + 1] += err * 1 / 16
    return img


def normalize_for_mnist(image: np.ndarray) -> np.ndarray:
    """Invert, apply Gaussian blur for anti-aliasing, and reshape to (1, 28, 28, 1) float32.

    The blur converts the binary/dithered image into grayscale values similar
    to anti-aliased MNIST digits. Uses a 3x3 Gaussian kernel with sigma≈0.7.
    """
    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ], dtype=np.float32) / 16.0
    h, w = image.shape
    padded = np.pad(image, 1, mode='constant', constant_values=0)
    blurred = np.empty((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            blurred[y, x] = np.sum(padded[y:y+3, x:x+3] * kernel)
    return (1.0 - blurred).reshape(1, 28, 28, 1)


def pbm_to_input(data: bytes) -> np.ndarray:
    """Convert 8x8 P4 PBM pixel data to (1, 28, 28, 1) float32 for the classifier."""
    grid = decode_pbm(data)
    upscaled = upscale(grid, 28, 28) # 28x28 for MNIST
    dithered = dither_binary(upscaled)
    return normalize_for_mnist(dithered)
