# Telephone Game — NATS Video Pipeline

A real-time video pipeline implementing the "telephone game" with neural networks. Digits are generated, degraded, classified, and re-generated through multiple stages — each stage "hears" the previous one's 8x8 binary output and "re-pronounces" its own 28x28 grayscale version through a CVAE generator.

## Models

- **`mnist_model.onnx`** — Fully-connected neural network trained on MNIST (90%+ accuracy on clean data, but sees degraded upscaled input)
- **`cvae_generator.onnx`** — Conditional VAE generator trained on MNIST. Takes `(latent_noise[16], label[10])` → outputs `28x28 grayscale`

### Custom Protocol

Each NATS/websocket message contains:
```
[PBM 8x8: 14 bytes][orig_size: 4 bytes BE uint32][orig_28x28: 784 bytes]
```

The original 28x28 grayscale image is preserved through the chain so the final display can compare input vs output.

### Files

| File | Description |
|------|-------------|
| `publish.py` | Generates digits 0-9 via CVAE, downscales to 8x8 PBM, publishes to NATS "one" |
| `classifier_cvae.py` | Classifies 8x8 → generates new 28x28 via CVAE → publishes 8x28 PBM to NATS "two" |
| `pbm_stream_to_ffmpeg.py` | Receives websocket frames, classifies + generates, displays side-by-side at 56x28 |
| `ws_bridge.py` | Bridges NATS subject to WebSocket (port 4195) |
| `digits.py` | Shared digit templates and PBM utilities |
| `start_telephone_game.sh` | Starts all backend processes (NATS, publisher, classifier, ws_bridge) |

## Quick Start

### 0. Dependencies

`uv`

```bash
./nats-server -js
nats str add one --subjects "one" --defaults
nats str info one -j | jq > one.config
```
change `"max_msgs": -1,` to `"max_msgs": 10,` in `one.config`

```bash
nats str rm one -f
nats str add one --config one.config
```

stop nats server.

### 1. Start the backend
```bash
./start_telephone_game.sh
```

This starts:
- NATS server (port 4222)
- ws_bridge (NATS "two" → WebSocket port 4195)
- Publisher (generates digits → NATS "one")
- Classifier (classifies → NATS "two")

### 2. Watch the telephone game
In a separate terminal:
```bash
uv run pbm_stream_to_ffmpeg.py
```

or record a video
```bash
uv run pbm_stream_to_ffmpeg.py out.mp4
```

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        START                                        │
│                                                                     │
│  Publisher (publish.py)                                             │
│  ├─ CVAE generates 28x28 grayscale digit (unique each time)         │
│  ├─ Downscales to 8x8 binary PBM (the "whisper")                    │
│  └─ Publishes to NATS "one" (PBM + original 28x28 embedded)         │
│                                                                     │
│  Classifier (classifier_cvae.py)                                    │
│  ├─ Subscribes to NATS "one"                                        │
│  ├─ Upscales 8x8 → 28x28, classifies with MNIST model               │
│  ├─ Generates new 28x28 from CVAE (predicted label + noise)         │
│  ├─ Downscales to 8x8 PBM                                           │
│  └─ Publishes to NATS "two" (PBM + original 28x28 passed through)   │
│                                                                     │
│  ws_bridge (ws_bridge.py)                                           │
│  └─ Subscribes to NATS "two", fans out to WebSocket clients         │
│                                                                     │
│  Display (pbm_stream_to_ffmpeg.py)                                  │
│  ├─ Connects to WebSocket                                           │
│  ├─ Classifies 8x8, generates 28x28 via CVAE                        │
│  ├─ Composites: original (left) vs generated (right) → 56x28        │
│  └─ Sends to ffplay or encodes to MP4                               │
│                                                                     │
│                         END                                         │
└─────────────────────────────────────────────────────────────────────┘
```
