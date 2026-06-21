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
LATENT_DIM = 16 # depends on cvae3_generator
GRID_SIZE = 40
IMGS_PER_FRAME = GRID_SIZE * GRID_SIZE
FPS = 4
CYCLES = 10  # Number of times to cycle through all labels
VIDEO_MODE = "random"  # "cycle" = sequential labels 0-11, "random" = random label each frame
TARGET_BRIGHTNESS = 32  # Target average brightness (0-255)
RESOLUTION = 720  # Output resolution (square)

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

            # Remove 1-pixel border (28x28 -> 26x26)
            img_cropped = img[1:-1, 1:-1]

            img_resized = cv2.resize(
                (img_cropped * 255).astype(np.uint8),
                (cell_size, cell_size),
                interpolation=cv2.INTER_NEAREST,
            )
            y_start = i * cell_size
            x_start = j * cell_size
            grid[y_start:y_start + cell_size, x_start:x_start + cell_size] = img_resized

    output_path = OUTPUT_DIR / f"grid_label_{label:02d}.png"
    cv2.imwrite(str(output_path), grid)
    print(f"  Saved: {output_path}")

# Cycling video
# Total duration = CYCLES * NUM_CLASSES / FPS seconds
OUTPUT_CYCLE = Path(__file__).parent / "output_cvae3_cycle.mp4"

print(f"\nWriting cycling video: {OUTPUT_CYCLE}")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer_cycle = cv2.VideoWriter(str(OUTPUT_CYCLE), fourcc, FPS, (RESOLUTION, RESOLUTION))

BATCH_SIZE = 64  # Process this many images at a time to limit memory

total_frames = CYCLES * NUM_CLASSES
print(f"Generating {total_frames} frames (mode: {VIDEO_MODE})...")

for frame_idx in range(total_frames):
    if VIDEO_MODE == "cycle":
        label = frame_idx % NUM_CLASSES
    else:  # random
        label = np.random.randint(0, NUM_CLASSES)

    # Generate one frame's worth of images in a small batch
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
            # Remove 1-pixel border (28x28 -> 26x26)
            img_cropped = img[1:-1, 1:-1]
            img_resized = cv2.resize(
                (img_cropped * 255).astype(np.uint8),
                (cell_size, cell_size),
                interpolation=cv2.INTER_NEAREST,
            )
            y_start = i * cell_size
            x_start = j * cell_size
            grid[y_start:y_start + cell_size, x_start:x_start + cell_size] = img_resized

    frame = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)

    # Global brightness scaling (only if too bright)
    current_mean = np.mean(frame)
    if current_mean > TARGET_BRIGHTNESS:
        scale = TARGET_BRIGHTNESS / current_mean
        frame = np.clip(frame * scale, 0, 255).astype(np.uint8)

    writer_cycle.write(frame)

    if (frame_idx + 1) % 10 == 0:
        print(f"  Generated {frame_idx + 1}/{total_frames} frames...")

writer_cycle.release()
print(f"Saved cycling video with {total_frames} frames to {OUTPUT_CYCLE}")
print("Done!")
