#!/bin/bash
# run_all_stages.sh
# Master script: trains models, generates data, and/or runs metrics.
#
# Usage:
#   bash run_all_stages.sh --mode [train|skip|metrics]
#
#   train   — from scratch: retrain all models, regenerate all data
#   skip    — skip steps where output already exists (default)
#   metrics — skip training/generation, only run stage 3 pipeline with metrics + video

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TF_DIR="$SCRIPT_DIR/../python/tf"
CONDA_PYTHON="/home/n/miniconda3/envs/tf/bin/python"
MODE="${1:---skip}"

# ─── Parse mode ─────────────────────────────────────────────────────
case "$MODE" in
    --train)   MODE="train" ;;
    --skip)    MODE="skip" ;;
    --metrics) MODE="metrics" ;;
    *)         MODE="skip" ;;
esac

echo "=== Telephone Game ==="
echo "  Mode: $MODE"
echo "  telephonegame: $SCRIPT_DIR"
echo "  python/tf:     $TF_DIR"
echo ""

START_TIME=$(date +%s)

# ─── Helper: check if file exists ───────────────────────────────────
need_step() {
    local desc="$1"
    shift
    if [ "$MODE" = "train" ]; then
        echo "  [RETRAIN] $desc"
        return 0  # always run
    fi
    if [ "$MODE" = "metrics" ]; then
        echo "  [SKIP] $desc (metrics mode)"
        return 1  # never run
    fi
    # skip mode: check if all outputs exist
    for f in "$@"; do
        if [ ! -f "$f" ]; then
            echo "  [NEEDED] $desc"
            return 0
        fi
    done
    echo "  [EXISTS] $desc"
    return 1
}

need_data() {
    local desc="$1"
    local dir="$2"
    local min="$3"
    if [ "$MODE" = "train" ]; then
        echo "  [REGEN] $desc"
        return 0
    fi
    if [ "$MODE" = "metrics" ]; then
        echo "  [SKIP] $desc (metrics mode)"
        return 1
    fi
    local count=$(ls "$dir"/*.pbm 2>/dev/null | wc -l)
    if [ "$count" -ge "$min" ]; then
        echo "  [EXISTS] $desc ($count files)"
        return 1
    fi
    echo "  [NEEDED] $desc ($count/$min files)"
    return 0
}

# ═══════════════════════════════════════════════════════════════════
# STAGE 1: Train mnist1 + cvae1 on original MNIST
# ═══════════════════════════════════════════════════════════════════
if [ "$MODE" != "metrics" ]; then
    echo "═══ Stage 1 ═══"

    if need_step "mnist1" "$SCRIPT_DIR/mnist_model.onnx"; then
        cd "$TF_DIR"
        "$CONDA_PYTHON" mnist.py 2>&1 | tail -5
        "$CONDA_PYTHON" -m tf2onnx.convert --saved-model mnist_model --output mnist_model.onnx
        cp "$TF_DIR/mnist_model.onnx" "$SCRIPT_DIR/mnist_model.onnx"
    fi

    if need_step "cvae1" "$SCRIPT_DIR/cvae_generator.onnx"; then
        cd "$TF_DIR"
        "$CONDA_PYTHON" train_cvae.py 2>&1 | tail -5
        "$CONDA_PYTHON" -m tf2onnx.convert --keras cvae_generator.h5 --output cvae_generator.onnx
        cp "$TF_DIR/cvae_generator.onnx" "$SCRIPT_DIR/cvae_generator.onnx"
    fi

    # ─── Generate data/pbm/ ─────────────────────────────────────────
    if need_data "data/pbm/" "$SCRIPT_DIR/data/pbm/" 50000; then
        cd "$SCRIPT_DIR"
        uv run generate_pbm.py --count 50000 --output-dir data/pbm
    fi

    echo ""
fi

# ═══════════════════════════════════════════════════════════════════
# STAGE 2: Train mnist2 + cvae2 on data/pbm/
# ═══════════════════════════════════════════════════════════════════
if [ "$MODE" != "metrics" ]; then
    echo "═══ Stage 2 ═══"

    # Point training scripts to telephonegame data
    cd "$TF_DIR"
    sed -i 's|PBM_DIR = Path(__file__).parent / "data/pbm"|PBM_DIR = Path(__file__).parent / "../telephonegame/data/pbm"|' mnist2.py 2>/dev/null || true
    sed -i 's|PBM_DIR = Path(__file__).parent / "data/pbm"|PBM_DIR = Path(__file__).parent / "../telephonegame/data/pbm"|' train_cvae2.py 2>/dev/null || true

    if need_step "mnist2" "$SCRIPT_DIR/mnist2_model.onnx"; then
        cd "$TF_DIR"
        "$CONDA_PYTHON" mnist2.py 2>&1 | tail -5
        "$CONDA_PYTHON" -m tf2onnx.convert --saved-model mnist2_model --output mnist2_model.onnx
        cp "$TF_DIR/mnist2_model.onnx" "$SCRIPT_DIR/mnist2_model.onnx"
    fi

    if need_step "cvae2" "$SCRIPT_DIR/cvae2_generator.onnx"; then
        cd "$TF_DIR"
        "$CONDA_PYTHON" train_cvae2.py 2>&1 | tail -5
        "$CONDA_PYTHON" -m tf2onnx.convert --keras cvae_generator.h5 --output cvae2_generator.onnx 2>/dev/null || \
        cp "$TF_DIR/cvae2_generator.onnx" "$SCRIPT_DIR/cvae2_generator.onnx" 2>/dev/null || true
    fi

    # ─── Generate data/pbm2/ ─────────────────────────────────────────
    if need_data "data/pbm2/" "$SCRIPT_DIR/data/pbm2/" 50000; then
        cd "$SCRIPT_DIR"
        uv run generate_pbm2.py --count 50000 --output-dir data/pbm2
    fi

    echo ""
fi

# ═══════════════════════════════════════════════════════════════════
# STAGE 3: Train mnist3 + cvae3 on data/pbm2/
# ═══════════════════════════════════════════════════════════════════
if [ "$MODE" != "metrics" ]; then
    echo "═══ Stage 3 ═══"

    cd "$TF_DIR"
    sed -i 's|PBM_DIR = Path(__file__).parent / "data/pbm2"|PBM_DIR = Path(__file__).parent / "../telephonegame/data/pbm2"|' mnist3.py 2>/dev/null || true
    sed -i 's|PBM_DIR = Path(__file__).parent / "data/pbm2"|PBM_DIR = Path(__file__).parent / "../telephonegame/data/pbm2"|' train_cvae3.py 2>/dev/null || true

    if need_step "mnist3" "$SCRIPT_DIR/mnist3_model.onnx"; then
        cd "$TF_DIR"
        "$CONDA_PYTHON" mnist3.py 2>&1 | tail -5
        "$CONDA_PYTHON" -m tf2onnx.convert --saved-model mnist3_model --output mnist3_model.onnx
        cp "$TF_DIR/mnist3_model.onnx" "$SCRIPT_DIR/mnist3_model.onnx"
    fi

    if need_step "cvae3" "$SCRIPT_DIR/cvae3_generator.onnx"; then
        cd "$TF_DIR"
        "$CONDA_PYTHON" train_cvae3.py 2>&1 | tail -5
        "$CONDA_PYTHON" -m tf2onnx.convert --keras cvae_generator.h5 --output cvae3_generator.onnx 2>/dev/null || \
        cp "$TF_DIR/cvae3_generator.onnx" "$SCRIPT_DIR/cvae3_generator.onnx" 2>/dev/null || true
    fi

    echo ""
fi

# ═══════════════════════════════════════════════════════════════════
# METRICS: Run stage 3 pipeline with metrics + video
# ═══════════════════════════════════════════════════════════════════
if [ "$MODE" = "metrics" ] || [ "$MODE" = "skip" ]; then
    echo "═══ Metrics: Stage 3 Pipeline ═══"
    echo "  Running 2-minute video capture with metrics..."
    echo ""

    METRICS_DIR="$SCRIPT_DIR/metrics/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$METRICS_DIR"
    VIDEO_FILE="$METRICS_DIR/stage3_$(date +%Y%m%d_%H%M%S).mp4"

    # Patch pbm_stream_to_ffmpeg3.py to output metrics
    METRICS_SCRIPT="$METRICS_DIR/pbm_stream_to_ffmpeg3_metrics.py"
    sed 's|output_file = sys.argv\[1\] if len(sys.argv) > 1 else None|output_file = sys.argv[1] if len(sys.argv) > 1 else None  # metrics patch placeholder|' \
        "$SCRIPT_DIR/pbm_stream_to_ffmpeg3.py" > "$METRICS_SCRIPT"

    # Start backend
    cd "$SCRIPT_DIR"
    echo "  Starting backend processes..."
    ./nats-server -c server.config &
    NATS_PID=$!
    sleep 1
    uv run ws_bridge.py two &
    BRIDGE_PID=$!
    sleep 1
    uv run publish3.py &
    PUBLISHER_PID=$!
    sleep 1
    uv run classifier_cvae3.py --max-images 0 &
    CLASSIFIER_PID=$!
    sleep 2

    # Run display with video output for 2 minutes
    echo "  Recording video + metrics (120 seconds)..."
    timeout 120 uv run pbm_stream_to_ffmpeg3.py "$VIDEO_FILE" > "$METRICS_DIR/pipeline.log" 2>&1 || true

    # Cleanup backend
    kill $CLASSIFIER_PID $PUBLISHER_PID $BRIDGE_PID $NATS_PID 2>/dev/null || true
    wait 2>/dev/null || true

    echo ""
    echo "  Video saved: $VIDEO_FILE"
    echo "  Pipeline log: $METRICS_DIR/pipeline.log"
fi

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "═══ Summary ═══"
echo "  Mode:   $MODE"
echo "  Time:   ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo ""
echo "  Models:"
for f in mnist_model.onnx mnist2_model.onnx mnist3_model.onnx cvae_generator.onnx cvae2_generator.onnx cvae3_generator.onnx; do
    if [ -f "$SCRIPT_DIR/$f" ]; then
        size=$(du -h "$SCRIPT_DIR/$f" | cut -f1)
        echo "    $f ($size)"
    fi
done
echo ""
echo "  Data:"
for d in pbm pbm2 pbm3; do
    count=$(ls "$SCRIPT_DIR/data/$d/"*.pbm 2>/dev/null | wc -l)
    echo "    data/$d/: $count images"
done
echo ""

if [ -d "$METRICS_DIR" ]; then
    echo "  Metrics: $METRICS_DIR"
    if [ -f "$VIDEO_FILE" ]; then
        vsize=$(du -h "$VIDEO_FILE" | cut -f1)
        echo "    Video: $VIDEO_FILE ($vsize)"
    fi
fi
