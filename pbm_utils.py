"""Shared PBM parsing and conversion utilities."""

import numpy as np

PBM_HEADER = b"P4\n8 8\n"
HEADER_LEN = len(PBM_HEADER)
PBM_DATA_BYTES = 8
PBM_TOTAL = HEADER_LEN + PBM_DATA_BYTES  # 14 bytes


def parse_message(data: bytes):
    """Parse a message into (publisher_digit, predicted, confidence, pbm_pixel_data, orig_bytes, width, height).

    Returns (None, None, None, None, None, None, None) if invalid.
    Format: [publisher_digit: 1 byte][predicted: 1 byte][confidence: 2 bytes BE uint16 / 10000][PBM...][orig...]
    """
    if len(data) < 4:
        return None, None, None, None, None, None, None

    publisher_digit = data[0]
    predicted = data[1]
    confidence = int.from_bytes(data[2:4], "big") / 10000.0
    rest = data[4:]

    p4_idx = rest.find(b"P4\n")
    if p4_idx == -1:
        return None, None, None, None, None, None, None

    try:
        header_end = rest.index(b"\n", p4_idx + 3)
        dims = rest[p4_idx + 3 : header_end].decode("ascii").strip().split()
        width, height = int(dims[0]), int(dims[1])
    except (ValueError, IndexError):
        return None, None, None, None, None, None, None

    header_len = header_end + 1
    bytes_per_row = (width + 7) // 8
    pbm_data_bytes = bytes_per_row * height

    idx = p4_idx
    pixel_data = rest[idx + header_len : idx + header_len + pbm_data_bytes]
    if len(pixel_data) < pbm_data_bytes:
        return None, None, None, None, None, None, None

    orig_start = idx + header_len + pbm_data_bytes
    if len(rest) < orig_start + 4:
        return publisher_digit, predicted, confidence, pixel_data, None, width, height
    orig_size = int.from_bytes(rest[orig_start : orig_start + 4], "big")
    orig_data = rest[orig_start + 4 : orig_start + 4 + orig_size]
    if len(orig_data) < orig_size:
        return publisher_digit, predicted, confidence, pixel_data, None, width, height

    return publisher_digit, predicted, confidence, pixel_data, orig_data, width, height


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
            grid[row_idx, col_idx] = 1.0 - bit  # invert: P4 black=1 -> white digit=1
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
    """Apply Gaussian blur for anti-aliasing and reshape to (1, 28, 28, 1) float32.

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
    #return blurred.reshape(1, 28, 28, 1)
    return image.reshape(1, 28, 28, 1)


def pbm_to_input(data: bytes, width: int = 8, height: int = 8) -> np.ndarray:
    """Convert P4 PBM pixel data to (1, 28, 28, 1) float32 for the old mnist model."""
    grid = decode_pbm(data, width=width, height=height)
    upscaled = upscale(grid, 28, 28)  # 28x28 for MNIST
    return normalize_for_mnist(upscaled)


def pbm_to_input2(data: bytes, width: int = 8, height: int = 8) -> np.ndarray:
    """Convert P4 PBM pixel data to (1, 28, 28) float32 for the mnist2 model.

    Upscales using nearest-neighbor (same as original pipeline).
    Returns binary image (0.0 or 1.0), no blur.
    """
    grid = decode_pbm(data, width=width, height=height)
    upscaled = upscale(grid, 28, 28)
    # Binarize and reshape to (1, 28, 28) — no channel dimension, no blur
    binary = (upscaled > 0.5).astype(np.float32)
    return binary.reshape(1, 28, 28)


def downscale_to_pbm(image_28x28: np.ndarray, width: int = 8, height: int = 8) -> bytes:
    """Downscale a 28x28 float32 image to a binary PBM of the given size.

    Uses the inverse of the upscale offset math: each destination pixel maps
    back to the corresponding block in the source, then thresholds at 0.5.
    """
    src_h, src_w = image_28x28.shape
    grid = np.zeros((height, width), dtype=np.float32)
    base_h, extra_h = divmod(src_h, height)
    base_w, extra_w = divmod(src_w, width)
    for r in range(height):
        for c in range(width):
            r_start = r * base_h + min(r, extra_h)
            c_start = c * base_w + min(c, extra_w)
            r_end = min(r_start + base_h + (1 if r < extra_h else 0), src_h)
            c_end = min(c_start + base_w + (1 if c < extra_w else 0), src_w)
            grid[r, c] = image_28x28[r_start:r_end, c_start:c_end].mean()

    binary = (grid > 0.5).astype(np.uint8)

    header = f"P4\n{width} {height}\n".encode("ascii")
    buf = bytearray(header)
    bytes_per_row = (width + 7) // 8
    for row in binary:
        val = 0
        for j in range(width):
            val = (val << 1) | (int(row[j]) & 1)
        val <<= (bytes_per_row * 8 - width)
        buf.extend(val.to_bytes(bytes_per_row, "big"))
    return bytes(buf)
