"""Train GAN and/or DCGAN.

Examples:
    # your own folders (cat/ and dog/ subfolders)
    python gan/train_gan.py --dataset folder --data data/train data/val --model both
    # AFHQ cat faces from Hugging Face (best quality)
    python gan/train_gan.py --dataset afhq --class_name cat --model dcgan --epochs 150 --batch_size 64
    # full Oxford-IIIT Pets, dogs only
    python gan/train_gan.py --dataset oxford_pets --class_name dog --model both --epochs 150 --batch_size 64
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gan_models import load_hf_images, train  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["folder", "afhq", "oxford_pets"], default="folder")
ap.add_argument("--data", nargs="+", help="folder(s) with cat/ and dog/ subfolders (for --dataset folder)")
ap.add_argument("--model", choices=["gan", "dcgan", "both"], default="both")
ap.add_argument("--class_name", default="all", help="'cat', 'dog' or 'all'")
ap.add_argument("--max_images", type=int, default=None)
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--sample_every", type=int, default=25)
ap.add_argument("--out_dir", default="gan_outputs")
ap.add_argument("--model_dir", default="model")
a = ap.parse_args()

if a.dataset == "folder":
    if not a.data:
        ap.error("--data is required with --dataset folder")
    data = a.data
else:
    data = load_hf_images(a.dataset, a.class_name, a.max_images)

for kind in (["gan", "dcgan"] if a.model == "both" else [a.model]):
    train(kind, data, a.class_name, a.epochs, a.batch_size, a.sample_every, a.out_dir, a.model_dir,
          dataset_name=a.dataset)
