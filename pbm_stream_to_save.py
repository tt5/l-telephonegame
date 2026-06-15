def save_pbm(pixel_data, width, height, predicted_digit, confidence, low_confidence=False):
    """Save PBM image to data/pbm directory.

    Filename format: {predicted_digit}_{confidence}[_low_conf].pbm
    """
    import time
    base = Path("data/pbm")
    base.mkdir(parents=True, exist_ok=True)

    header = f"P4\n{width} {height}\n".encode("ascii")
    pbm_bytes = header + bytes(pixel_data)
    conf_str = f"{confidence:.4f}"
    suffix = "_low_conf" if low_confidence else ""
    filename = f"{predicted_digit}_{conf_str}{suffix}_{int(time.time()*1000)}.pbm"
    (base / filename).write_bytes(pbm_bytes)
