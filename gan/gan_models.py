"""GAN and DCGAN for generating cat / dog images (TensorFlow / Keras).

- Vanilla GAN : fully-connected (Dense) generator and discriminator.
- DCGAN       : convolutional generator (upsampling convolutions) and discriminator (Conv2D),
                following Radford et al. 2015 (BatchNorm, LeakyReLU, Adam lr=2e-4, beta1=0.5).

Images are 64x64 RGB scaled to [-1, 1] (generator ends in tanh).
Data: your own folders, or larger public datasets from Hugging Face (no login needed):
  - AFHQ (Animal Faces-HQ): ~5k aligned cat faces, ~5k aligned dog faces  -> best results
  - Oxford-IIIT Pets: ~2.4k cat and ~4.9k dog photos (whole animal, varied poses)
Real AND fake images are lightly augmented (flip + shift) before the discriminator sees them
("differentiable augmentation"), and the saved generator is an EMA of the weights.
"""
import glob
import json
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 64
CHANNELS = 3
LATENT_DIM = 100
DCGAN_UPSAMPLE = "resize"   # 'resize' (no checkerboard) or 'transpose' (original paper)


# --------------------------------------------------------------------------- data
def load_dataset(data_dirs, class_name="all", batch_size=32):
    """Load images from one or more folders laid out as <dir>/<class>/*.jpg.

    class_name: 'cat', 'dog' (folder name) or 'all'.
    Returns (tf.data.Dataset of batches in [-1, 1], number of images).
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    files = []
    for d in data_dirs:
        sub = [class_name] if class_name != "all" else sorted(os.listdir(d))
        for c in sub:
            for e in exts:
                files += glob.glob(os.path.join(d, c, e))
    if not files:
        raise FileNotFoundError(f"No images found for class '{class_name}' in {data_dirs}")

    def load(path):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), antialias=True)
        return tf.cast(img, tf.float32) / 127.5 - 1.0

    ds = (tf.data.Dataset.from_tensor_slices(files)
          .map(load, num_parallel_calls=tf.data.AUTOTUNE)
          .cache()
          .shuffle(len(files))
          .batch(batch_size, drop_remainder=True)
          .prefetch(tf.data.AUTOTUNE))
    return ds, len(files)


HF_DATASETS = {
    # name: (Hugging Face repo, label column, splits)
    "afhq": ("huggan/AFHQ", "label", ["train"]),                       # 512px aligned faces: cat / dog / wild
    "oxford_pets": ("timm/oxford-iiit-pet", "label_cat_dog", ["train", "test"]),  # 7.3k pet photos: cat / dog
}


def _square_resize(img, size=IMG_SIZE):
    """Centre-crop to a square, then resize (keeps the animal centred, no stretching)."""
    from PIL import Image
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return np.asarray(img.resize((size, size), Image.LANCZOS), dtype="uint8")


def load_hf_images(dataset="afhq", class_name="cat", max_images=None, cache_dir="data_cache"):
    """Download a public Hugging Face dataset (no login needed), keep one class, and
    return a uint8 array (N, 64, 64, 3). The result is cached as .npy for re-runs."""
    from datasets import load_dataset
    repo, label_col, splits = HF_DATASETS[dataset]
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{dataset}_{class_name}_{IMG_SIZE}.npy")
    if os.path.exists(cache):
        arr = np.load(cache)
        print(f"Loaded {len(arr)} cached images from {cache}")
        return arr[:max_images] if max_images else arr
    imgs = []
    for split in splits:
        ds = load_dataset(repo, split=split)
        if class_name != "all":
            names = ds.features[label_col].names
            if class_name not in names:
                raise ValueError(f"class_name must be one of {names} or 'all'")
            want = names.index(class_name)
            ds = ds.filter(lambda labels: [l == want for l in labels], input_columns=[label_col],
                           batched=True)
        print(f"{repo} [{split}] -> {len(ds)} '{class_name}' images, resizing to {IMG_SIZE}x{IMG_SIZE}...")
        for ex in ds:
            imgs.append(_square_resize(ex["image"]))
            if max_images and len(imgs) >= max_images:
                break
    arr = np.stack(imgs)
    np.save(cache, arr)
    print(f"Saved {len(arr)} images to {cache}")
    return arr


def dataset_from_array(arr, batch_size=64):
    """uint8 (N, H, W, 3) array -> shuffled tf.data batches in [-1, 1]."""
    ds = (tf.data.Dataset.from_tensor_slices(arr)
          .shuffle(len(arr))
          .map(lambda x: tf.cast(x, tf.float32) / 127.5 - 1.0, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(batch_size, drop_remainder=True)
          .prefetch(tf.data.AUTOTUNE))
    return ds, len(arr)


def diff_augment(x):
    """Random horizontal flip + random shift (same ops for real and fake, differentiable)."""
    flip = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) < 0.5
    x = tf.where(flip, tf.image.flip_left_right(x), x)
    pad = 6
    x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
    x = tf.image.random_crop(x, tf.stack([tf.shape(x)[0], IMG_SIZE, IMG_SIZE, CHANNELS]))
    return x


# ----------------------------------------------------------------------- models
def build_gan_generator(latent_dim=LATENT_DIM):
    return models.Sequential([
        layers.Input((latent_dim,)),
        layers.Dense(256), layers.LeakyReLU(0.2), layers.BatchNormalization(),
        layers.Dense(512), layers.LeakyReLU(0.2), layers.BatchNormalization(),
        layers.Dense(1024), layers.LeakyReLU(0.2), layers.BatchNormalization(),
        layers.Dense(IMG_SIZE * IMG_SIZE * CHANNELS, activation="tanh"),
        layers.Reshape((IMG_SIZE, IMG_SIZE, CHANNELS)),
    ], name="gan_generator")


def build_gan_discriminator():
    return models.Sequential([
        layers.Input((IMG_SIZE, IMG_SIZE, CHANNELS)),
        layers.Flatten(),
        layers.Dense(1024), layers.LeakyReLU(0.2), layers.Dropout(0.3),
        layers.Dense(512), layers.LeakyReLU(0.2), layers.Dropout(0.3),
        layers.Dense(256), layers.LeakyReLU(0.2), layers.Dropout(0.3),
        layers.Dense(1),  # logit
    ], name="gan_discriminator")


def _up_block(filters, upsample):
    """2x upsampling block. 'transpose' = classic DCGAN (Conv2DTranspose);
    'resize' = nearest-neighbour upsample + Conv2D, which avoids checkerboard artifacts
    (Odena et al., 2016) and works better on small datasets."""
    if upsample == "transpose":
        return [layers.Conv2DTranspose(filters, 4, strides=2, padding="same", use_bias=False)]
    return [layers.UpSampling2D(2, interpolation="nearest"),
            layers.Conv2D(filters, 3, padding="same", use_bias=False)]


def build_dcgan_generator(latent_dim=LATENT_DIM, upsample=None):
    # 100 -> 4x4x512 -> 8x8x256 -> 16x16x128 -> 32x32x64 -> 64x64x3
    upsample = upsample or DCGAN_UPSAMPLE
    blocks = []
    for f in (256, 128, 64):
        blocks += _up_block(f, upsample) + [layers.BatchNormalization(), layers.ReLU()]
    if upsample == "transpose":
        out = [layers.Conv2DTranspose(CHANNELS, 4, strides=2, padding="same", activation="tanh")]
    else:
        out = [layers.UpSampling2D(2, interpolation="nearest"),
               layers.Conv2D(CHANNELS, 3, padding="same", activation="tanh")]
    return models.Sequential([
        layers.Input((latent_dim,)),
        layers.Dense(4 * 4 * 512, use_bias=False), layers.BatchNormalization(), layers.ReLU(),
        layers.Reshape((4, 4, 512)),
        *blocks, *out,
    ], name="dcgan_generator")


def build_dcgan_discriminator():
    # 64x64x3 -> 32 -> 16 -> 8 -> 4 -> logit   (no BatchNorm on the first layer, as in the paper)
    return models.Sequential([
        layers.Input((IMG_SIZE, IMG_SIZE, CHANNELS)),
        layers.Conv2D(64, 4, strides=2, padding="same"), layers.LeakyReLU(0.2),
        layers.Conv2D(128, 4, strides=2, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.LeakyReLU(0.2),
        layers.Conv2D(256, 4, strides=2, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.LeakyReLU(0.2),
        layers.Conv2D(512, 4, strides=2, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.LeakyReLU(0.2),
        layers.Flatten(), layers.Dropout(0.3),
        layers.Dense(1),  # logit
    ], name="dcgan_discriminator")


BUILDERS = {
    "gan": (build_gan_generator, build_gan_discriminator),
    "dcgan": (build_dcgan_generator, build_dcgan_discriminator),
}


# ---------------------------------------------------------------------- helpers
def to_uint8(imgs):
    return np.clip((np.asarray(imgs) + 1.0) * 127.5, 0, 255).astype("uint8")


def save_grid(imgs, path, cols=8):
    """Save a batch of images in [-1, 1] as one PNG grid."""
    from PIL import Image
    imgs = to_uint8(imgs)
    n = len(imgs)
    rows = int(np.ceil(n / cols))
    grid = np.ones((rows * IMG_SIZE + (rows + 1) * 2, cols * IMG_SIZE + (cols + 1) * 2, 3), "uint8") * 255
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        y, x = 2 + r * (IMG_SIZE + 2), 2 + c * (IMG_SIZE + 2)
        grid[y:y + IMG_SIZE, x:x + IMG_SIZE] = im
    Image.fromarray(grid).resize((grid.shape[1] * 2, grid.shape[0] * 2), Image.NEAREST).save(path)


def make_gif(png_paths, gif_path, ms=400):
    from PIL import Image
    frames = [Image.open(p) for p in png_paths]
    if frames:
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=ms, loop=0)


# ---------------------------------------------------------------------- training
def train(kind, data, class_name="all", epochs=300, batch_size=32, sample_every=25,
          out_dir="gan_outputs", model_dir="model", latent_dim=LATENT_DIM, seed=42,
          ema_decay=0.999, dataset_name="folder"):
    """Train a GAN ('gan') or DCGAN ('dcgan').

    data: folder path(s) laid out as <dir>/<class>/*.jpg, OR a uint8 array (N, 64, 64, 3)
          (e.g. from load_hf_images).
    The saved generator is an EMA (exponential moving average) copy of the generator's
    weights, which gives smoother, cleaner samples than the raw weights.
    Saves the generator, sample grids, a progress GIF and loss curves.
    """
    assert kind in BUILDERS, "kind must be 'gan' or 'dcgan'"
    tf.keras.utils.set_random_seed(seed)
    if isinstance(data, np.ndarray):
        ds, n = dataset_from_array(data, batch_size)
    else:
        ds, n = load_dataset(data, class_name, batch_size)
    if n < batch_size:
        raise ValueError(f"Only {n} images but batch_size={batch_size}. Use a smaller batch_size.")
    print(f"[{kind}] {n} training images ({dataset_name}, class='{class_name}'), {epochs} epochs, "
          f"batch {batch_size}")

    G, D = BUILDERS[kind][0](latent_dim), BUILDERS[kind][1]()
    G_ema = tf.keras.models.clone_model(G)
    G_ema.set_weights(G.get_weights())
    g_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    d_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def train_step(real):
        b = tf.shape(real)[0]
        z = tf.random.normal([b, latent_dim])
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake = G(z, training=True)
            r_logit = D(diff_augment(real), training=True)
            f_logit = D(diff_augment(fake), training=True)
            # one-sided label smoothing (real = 0.9) keeps D from getting over-confident
            d_loss = bce(tf.ones_like(r_logit) * 0.9, r_logit) + bce(tf.zeros_like(f_logit), f_logit)
            g_loss = bce(tf.ones_like(f_logit), f_logit)
        g_opt.apply_gradients(zip(g_tape.gradient(g_loss, G.trainable_variables), G.trainable_variables))
        d_opt.apply_gradients(zip(d_tape.gradient(d_loss, D.trainable_variables), D.trainable_variables))
        for w_ema, w in zip(G_ema.weights, G.weights):  # EMA of all weights (incl. BatchNorm stats)
            w_ema.assign(ema_decay * w_ema + (1.0 - ema_decay) * w)
        return g_loss, d_loss

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    fixed_z = tf.random.normal([32, latent_dim], seed=seed)  # same noise every time -> see progress
    hist, grids = {"g_loss": [], "d_loss": []}, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        gl, dl, steps = 0.0, 0.0, 0
        for real in ds:
            g, d = train_step(real)
            gl, dl, steps = gl + float(g), dl + float(d), steps + 1
        hist["g_loss"].append(gl / steps)
        hist["d_loss"].append(dl / steps)
        elapsed = time.time() - t0
        is_sample = ep % sample_every == 0 or ep == 1 or ep == epochs
        if is_sample:
            p = os.path.join(out_dir, f"{kind}_epoch_{ep:04d}.png")
            save_grid(G_ema(fixed_z, training=False), p)
            grids.append(p)
        if is_sample or ep <= 3:
            # estimate from epochs after the first (epoch 1 includes one-time graph compilation)
            per_ep = (elapsed - first_ep) / (ep - 1) if ep > 1 else elapsed
            eta = per_ep * (epochs - ep)
            print(f"[{kind}] epoch {ep}/{epochs}  G={hist['g_loss'][-1]:.3f}  D={hist['d_loss'][-1]:.3f}"
                  f"  ({elapsed:.0f}s elapsed, ~{per_ep:.1f}s/epoch, ~{eta / 60:.0f} min left)")
            if ep == 2 and per_ep > 30 and not tf.config.list_physical_devices("GPU"):
                print("⚠️  Training on CPU is very slow. In Colab: Runtime → Change runtime type → T4 GPU.")
        if ep == 1:
            first_ep = elapsed

    G_ema.save(os.path.join(model_dir, f"{kind}_generator.keras"))
    with open(os.path.join(model_dir, f"{kind}_info.json"), "w") as f:
        json.dump({"kind": kind, "dataset": dataset_name, "class": class_name, "epochs": epochs,
                   "latent_dim": latent_dim, "images": n}, f)
    make_gif(grids, os.path.join(out_dir, f"{kind}_progress.gif"))
    save_grid(G_ema(fixed_z, training=False), os.path.join(out_dir, f"{kind}_final.png"))

    try:  # Figure API: saves the plot without changing the notebook's display backend
        from matplotlib.figure import Figure
        fig = Figure(figsize=(8, 4))
        ax = fig.subplots()
        ax.plot(hist["g_loss"], label="Generator")
        ax.plot(hist["d_loss"], label="Discriminator")
        ax.set_title(f"{kind.upper()} losses"); ax.set_xlabel("epoch"); ax.legend()
        fig.savefig(os.path.join(out_dir, f"{kind}_losses.png"), dpi=110, bbox_inches="tight")
    except ImportError:
        pass
    print(f"[{kind}] saved generator to {model_dir}/{kind}_generator.keras and samples to {out_dir}/")
    return G_ema, hist
