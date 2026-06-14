#!/usr/bin/env python3
"""test_model.py — Evaluate the ONNX MNIST model on real MNIST test data.

This tells us the baseline accuracy we should expect when the input is correct.
If this is high but the pipeline accuracy is low, the problem is in the pipeline.
If this is also low, the model itself is the problem.
"""

import urllib.request
import gzip
import struct
import numpy as np


def download_mnist(filename, url_base="https://storage.googleapis.com/cvdf-datasets/mnist/"):
    """Download MNIST file if not already cached."""
    import os
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        url = url_base + filename
        url = url.replace(".gz", "")  # Try without .gz first
        try:
            urllib.request.urlretrieve(url, filename)
        except Exception:
            # Try with .gz
            gz_filename = ".telephone_cache" + filename
            url_gz = url_base + filename
            urllib.request.urlretrieve(url_gz, gz_filename)
            with gzip.open(gz_filename, 'rb') as f_in:
                with open(filename, 'wb') as f_out:
                    f_out.write(f_in.read())
            os.remove(gz_filename)
    return filename


def read_mnist_images(filename):
    """Read MNIST image file into numpy array."""
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"Bad magic number {magic} in {filename}"
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return data


def read_mnist_labels(filename):
    """Read MNIST label file into numpy array."""
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        assert magic == 2049, f"Bad magic number {magic} in {filename}"
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def main():
    print("Loading ONNX model...")
    import onnxruntime as ort
    session = ort.InferenceSession("mnist_model.onnx")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"  Input:  {input_name} {session.get_inputs()[0].shape}")
    print(f"  Output: {output_name} {session.get_outputs()[0].shape}")

    print("\nLoading MNIST test data...")
    # Use keras to get the data (simpler than downloading manually)
    try:
        import tensorflow as tf
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    except ImportError:
        print("TensorFlow not available, trying to download MNIST manually...")
        import os, gzip, struct
        cache = ".mnist_cache"
        os.makedirs(cache, exist_ok=True)

        def maybe_download(name):
            path = os.path.join(cache, name)
            if not os.path.exists(path):
                gz_path = path + ".gz"
                url = f"https://storage.googleapis.com/cvdf-datasets/mnist/{name}.gz"
                print(f"  Downloading {name}.gz...")
                urllib.request.urlretrieve(url, gz_path)
                with gzip.open(gz_path, 'rb') as fin, open(path, 'wb') as fout:
                    fout.write(fin.read())
                os.remove(gz_path)
            return path

        x_test = read_mnist_images(maybe_download("t10k-images-idx3-ubyte"))
        y_test = read_mnist_labels(maybe_download("t10k-labels-idx1-ubyte"))

    print(f"  Test samples: {len(x_test)}")
    print(f"  Image shape:  {x_test[0].shape}")
    print(f"  Value range:  [{x_test.min()}, {x_test.max()}]")

    # Normalize to 0-1 float32 (same as training)
    x_test_float = x_test.astype(np.float32) / 255.0

    # Test with full batch
    correct = 0
    total = len(x_test)
    batch_size = 100

    print(f"\nEvaluating model on {total} test samples (batch_size={batch_size})...")
    for i in range(0, total, batch_size):
        batch = x_test_float[i:i+batch_size]
        # Reshape to (batch, 28, 28, 1) — same as what the pipeline produces
        batch_input = batch.reshape(-1, 28, 28, 1).astype(np.float32)
        outputs = session.run([output_name], {input_name: batch_input})[0]
        predictions = np.argmax(outputs, axis=1)
        labels = y_test[i:i+batch_size]
        correct += np.sum(predictions == labels)

        if (i // batch_size) % 10 == 0:
            print(f"  Progress: {min(i+batch_size, total)}/{total}")

    accuracy = correct / total
    print(f"\n{'='*50}")
    print(f"Model accuracy on MNIST test set: {correct}/{total} = {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*50}")

    if accuracy > 0.95:
        print("\nModel is performing correctly on real MNIST data.")
        print("If the pipeline accuracy is lower, the problem is in the pipeline")
        print("(PBM encoding, upscaling, blur, normalization, etc.)")
    else:
        print(f"\nModel accuracy is only {accuracy*100:.2f}% — lower than expected for MNIST.")
        print("The model itself may be the problem.")


if __name__ == "__main__":
    main()
