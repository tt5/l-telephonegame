#!/usr/bin/env python3
"""save_video.py

Generate images from cvae3_generator.onnx cycling through labels 0-11.
Saves 12 PNG images (one per label, GRID_SIZE x GRID_SIZE grid) and one cycling video.

Usage:
    python save_video.py
"""

import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "cvae3_generator.onnx"
OUTPUT_DIR = Path(__file__).parent / "output_grids"

NUM_CLASSES = 12
LATENT_DIM = 16
GRID_SIZE = 6
IMGS_PER_FRAME = GRID_SIZE * GRID_SIZE
FPS = 2
RESOLUTION = 640  # Output resolution (square)

print(f"Loading model: {MODEL_PATH}")
sess = ort.InferenceSession(str(MODEL_PATH))
latent_input_name = sess.get_inputs()[0].name
label_input_name = sess.get_inputs()[1].name
output_name = sess.get_outputs()[0].name

print(f"  Latent dim: {LATENT_DIM}")
print(f"  Num classes: {NUM_CLASSES}")
print(f"  Grid: {GRID_SIZE}x{GRID_SIZE} = {IMGS_PER_FRAME} images")

OUTPUT_DIR.mkdir(exist_ok=True)
cell_size = RESOLUTION // GRID_SIZE

for label in range(NUM_CLASSES):
    print(f"Generating label {label}...")

    noise = np.random.normal(size=(IMGS_PER_FRAME, LATENT_DIM)).astype(np.float32)
    labels = np.zeros((IMGS_PER_FRAME, NUM_CLASSES), dtype=np.float32)
    labels[:, label] = 1.0

    gen_images = sess.run(
        [output_name],
        {latent_input_name: noise, label_input_name: labels},
    )[0][:, :, :, 0]

    grid = np.zeros((RESOLUTION, RESOLUTION), dtype=np.uint8)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            idx = i * GRID_SIZE + j
            img = gen_images[idx]
            img_resized = cv2.resize(
                (img * 255).astype(np.uint8),
                (cell_size, cell_size),
                interpolation=cv2.INTER_NEAREST,
            )
            y_start = i * cell_size
            x_start = j * cell_size
            grid[y_start:y_start + cell_size, x_start:x_start + cell_size] = img_resized

    cv2.putText(
        grid,
        f"Label: {label}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    output_path = OUTPUT_DIR / f"grid_label_{label:02d}.png"
    cv2.imwrite(str(output_path), grid)
    print(f"  Saved: {output_path}")

# Cycling video
CYCLES = 3
OUTPUT_CYCLE = Path(__file__).parent / "output_cvae3_cycle.mp4"

print(f"\nWriting cycling video: {OUTPUT_CYCLE}")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer_cycle = cv2.VideoWriter(str(OUTPUT_CYCLE), fourcc, FPS, (RESOLUTION, RESOLUTION))

all_noise = np.random.normal(size=(NUM_CLASSES * CYCLES * IMGS_PER_FRAME, LATENT_DIM)).astype(np.float32)
all_labels = np.zeros((NUM_CLASSES * CYCLES * IMGS_PER_FRAME, NUM_CLASSES), dtype=np.float32)

idx = 0
for cycle in range(CYCLES):
    for label in range(NUM_CLASSES):
        for _ in range(IMGS_PER_FRAME):
            all_labels[idx, label] = 1.0
            idx += 1

print("Generating all images for cycle video...")
all_images = sess.run(
    [output_name],
    {latent_input_name: all_noise, label_input_name: all_labels},
)[0][:, :, :, 0]

frame_idx = 0
for cycle in range(CYCLES):
    for label in range(NUM_CLASSES):
        grid = np.zeros((RESOLUTION, RESOLUTION), dtype=np.uint8)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                img_idx = frame_idx * IMGS_PER_FRAME + i * GRID_SIZE + j
                img = all_images[img_idx]
                img_resized = cv2.resize(
                    (img * 255).astype(np.uint8),
                    (cell_size, cell_size),
                    interpolation=cv2.INTER_NEAREST,
                )
                y_start = i * cell_size
                x_start = j * cell_size
                grid[y_start:y_start + cell_size, x_start:x_start + cell_size] = img_resized

        frame = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            frame,
            f"Label: {label} (Cycle {cycle + 1}/{CYCLES})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        writer_cycle.write(frame)
        frame_idx += 1

writer_cycle.release()
print(f"Saved cycling video with {CYCLES * NUM_CLASSES} frames to {OUTPUT_CYCLE}")
print("Done!")
