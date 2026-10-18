"""Generate new images with a trained generator.

Examples:
    python generate.py                          # 16 DCGAN images -> generated.png
    python generate.py --model gan --n 32 --seed 7 --out gan_samples.png
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gan"))
from gan_models import save_grid  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["dcgan", "gan"], default="dcgan")
ap.add_argument("--model_dir", default="model")
ap.add_argument("--n", type=int, default=16, help="number of images")
ap.add_argument("--seed", type=int, default=42, help="same seed = same images")
ap.add_argument("--out", default="generated.png")
a = ap.parse_args()

path = os.path.join(a.model_dir, f"{a.model}_generator.keras")
if not os.path.exists(path):
    sys.exit(f"Generator not found: {path}. Train first (see README) and put the file in {a.model_dir}/.")

G = tf.keras.models.load_model(path, compile=False)
z = np.random.default_rng(a.seed).normal(size=(a.n, G.input_shape[-1])).astype("float32")
save_grid(G.predict(z, verbose=0), a.out, cols=min(8, a.n))
print(f"Saved {a.n} {a.model.upper()} images to {a.out}")
