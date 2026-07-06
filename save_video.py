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

NUM_CLASSES = 3+2
LATENT_DIM = 384 # depends on cvae3_generator
GRID_SIZE = 36
IMGS_PER_FRAME = GRID_SIZE * GRID_SIZE
FPS = 8
CYCLES = 30  # Number of times to cycle through all labels
VIDEO_MODE = "cycle"  # "cycle" = sequential, "random" = random, "mixed" = upper half cycle, lower half random
BRIGHTNESS_WINDOW = 1 # Number of frames to average over for dynamic brightness target
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

    grid = np.zeros((RESOLUTION, RESOLUTION), dtype=np.uint8)
    for i in range(GRID_SIZE):
        # Mixed mode: upper half = current label, lower half = random label
        if VIDEO_MODE == "mixed" and i >= GRID_SIZE // 2:
            row_label = np.random.randint(0, NUM_CLASSES)
        else:
            row_label = label

        noise = np.random.normal(size=(GRID_SIZE, LATENT_DIM)).astype(np.float32)
        labels = np.zeros((GRID_SIZE, NUM_CLASSES), dtype=np.float32)
        labels[:, row_label] = 1.0

        gen_images = sess.run(
            [output_name],
            {latent_input_name: noise, label_input_name: labels},
        )[0][:, :, :, 0]

        for j in range(GRID_SIZE):
            img = gen_images[j]
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

brightness_history = []  # Rolling window of last N frames' brightness

for frame_idx in range(total_frames):
    grid = np.zeros((RESOLUTION, RESOLUTION), dtype=np.uint8)

    # Determine labels for this frame
    if VIDEO_MODE == "cycle":
        frame_label = frame_idx % NUM_CLASSES
    elif VIDEO_MODE == "random":
        frame_label = np.random.randint(0, NUM_CLASSES)
    else:  # mixed
        cycle_label = frame_idx % NUM_CLASSES
        random_label = np.random.randint(0, NUM_CLASSES)

    for i in range(GRID_SIZE):
        if VIDEO_MODE == "mixed":
            label = cycle_label if i < GRID_SIZE // 2 else random_label
        else:
            label = frame_label

        # Generate one row's worth of images
        noise = np.random.normal(size=(GRID_SIZE, LATENT_DIM)).astype(np.float32)
        labels = np.zeros((GRID_SIZE, NUM_CLASSES), dtype=np.float32)
        labels[:, label] = 1.0

        gen_images = sess.run(
            [output_name],
            {latent_input_name: noise, label_input_name: labels},
        )[0][:, :, :, 0]

        for j in range(GRID_SIZE):
            img = gen_images[j]
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

    # Dynamic brightness scaling (only if too bright)
    current_mean = np.mean(frame)
    brightness_history.append(current_mean)
    if len(brightness_history) > BRIGHTNESS_WINDOW:
        brightness_history.pop(0)
    target = np.mean(brightness_history)
    if current_mean > target:
        scale = target / current_mean
        scale = (scale * scale) * 2
        if VIDEO_MODE == "mixed":
            scale = 1
        #scale = 1
        frame = np.clip(frame * scale, 0, 255).astype(np.uint8)

    writer_cycle.write(frame)

    if (frame_idx + 1) % 10 == 0:
        print(f"  Generated {frame_idx + 1}/{total_frames} frames...")

writer_cycle.release()
print(f"Saved cycling video with {total_frames} frames to {OUTPUT_CYCLE}")
print("Done!")
