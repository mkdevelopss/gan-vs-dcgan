"""Streamlit demo: generate new cat images with the trained GAN / DCGAN generators.

Run:  streamlit run app.py
"""
import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
LABELS = {"dcgan": "DCGAN (convolutional)", "gan": "Vanilla GAN (fully connected)"}
DATASETS = {"afhq": "AFHQ animal faces", "oxford_pets": "Oxford-IIIT Pets",
            "folder": "uploaded photos", "upload": "uploaded photos"}

st.set_page_config(page_title="GAN vs DCGAN", page_icon="🎨")
st.title("🎨 GAN vs DCGAN Image Generator")
st.write("Every image below is **made up by the generator** from 100 random numbers. "
         "None of them are real photos.")


@st.cache_resource(show_spinner="Loading generator...")
def load_generator(kind):
    g = tf.keras.models.load_model(os.path.join(MODEL_DIR, f"{kind}_generator.keras"), compile=False)
    info_path = os.path.join(MODEL_DIR, f"{kind}_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    return g, info


available = [k for k in LABELS if os.path.exists(os.path.join(MODEL_DIR, f"{k}_generator.keras"))]
if not available:
    found = sorted(os.listdir(MODEL_DIR)) if os.path.isdir(MODEL_DIR) else []
    st.info("No generator found. Train with the notebook or `gan/train_gan.py`, then put "
            "`dcgan_generator.keras` (and optionally `gan_generator.keras`) in the `model/` folder.")
    st.caption(f"Files in model/: {', '.join(found) if found else '(empty)'}")
    st.stop()

kind = st.radio("Model", available, format_func=LABELS.get, horizontal=True)
n = st.slider("How many images", 4, 32, 16, step=4)
if "seed" not in st.session_state:
    st.session_state["seed"] = 42
if st.button("🎲 Generate new images", type="primary"):
    st.session_state["seed"] = int(np.random.randint(0, 1_000_000))

G, info = load_generator(kind)
if info:
    st.caption(f"Trained on {DATASETS.get(info.get('dataset'), info.get('dataset', '?'))} · "
               f"{info.get('class', 'all')} ({info.get('images', '?')} images), "
               f"{info.get('epochs', '?')} epochs · seed {st.session_state['seed']}")

z = np.random.default_rng(st.session_state["seed"]).normal(size=(n, G.input_shape[-1])).astype("float32")
imgs = np.clip((G.predict(z, verbose=0) + 1) * 127.5, 0, 255).astype("uint8")
cols = st.columns(4)
for i, im in enumerate(imgs):
    cols[i % 4].image(Image.fromarray(im).resize((128, 128), Image.LANCZOS), width="stretch")
