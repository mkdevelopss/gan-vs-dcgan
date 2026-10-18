# 🎨 GAN vs DCGAN: Generating Cat Faces from Scratch

Two generative adversarial networks, a **vanilla GAN** (fully connected) and a **DCGAN** (convolutional), implemented in **TensorFlow/Keras** and trained **from scratch** on the same dataset to compare how architecture affects image quality.

Both models learn to turn **100 random numbers** into a new **64×64 cat face** that doesn't exist in the dataset.

<!-- After training, copy your images into assets/ (see assets/README.md) and they will show here. -->
![Real vs GAN vs DCGAN](assets/comparison.png)

| DCGAN samples | Training progress |
|---|---|
| ![DCGAN samples](assets/dcgan_final.png) | ![DCGAN progress](assets/dcgan_progress.gif) |

---

## How a GAN works

A GAN is two networks trained against each other:

- **Generator (G):** takes a random noise vector `z` (100 numbers) and produces an image.
- **Discriminator (D):** looks at an image and outputs how likely it is to be real.

Each training step:
1. D sees a batch of **real** images and a batch of G's **fake** images, and learns to tell them apart.
2. G is updated so that D is more likely to call its fakes "real". The gradient flows *through* D into G.

G never sees a real image directly; it learns only from D's feedback. After training, D is discarded and only G is used to generate images.

## Models

| | Vanilla GAN | DCGAN |
|---|---|---|
| Generator | Dense 256 → 512 → 1024 → 12,288, reshaped to 64×64×3 | Dense → 4×4×512 → (UpSampling + Conv2D) ×3 → 32×32×64 → UpSampling + Conv2D → 64×64×3 |
| Discriminator | Flatten → Dense 1024 → 512 → 256 → 1 | Conv2D (stride 2) 64 → 128 → 256 → 512 → Dense 1 |
| Normalization | BatchNorm in G | BatchNorm in G and D (not on D's first layer) |
| Activations | LeakyReLU, tanh output | ReLU in G, LeakyReLU(0.2) in D, tanh output |
| Understands spatial structure? | ❌ treats pixels as a flat list | ✅ convolutions learn local patterns (edges, eyes, fur) |

Both follow the DCGAN paper's training setup: **Adam (lr = 2e-4, β1 = 0.5)**, binary cross-entropy loss, latent size 100.

### Training tricks
- **Upsample + Conv2D instead of transposed convolutions:** removes checkerboard artifacts ([Odena et al., 2016](https://distill.pub/2016/deconv-checkerboard/)). Set `DCGAN_UPSAMPLE = "transpose"` for the original paper version.
- **Augmentation on real *and* fake images** (random flip + shift) so the discriminator can't memorise the training set.
- **One-sided label smoothing:** real labels are 0.9 instead of 1.0, so D doesn't become overconfident.
- **Dropout in D** to keep it from overpowering G.
- **EMA generator:** the saved generator is an exponential moving average of the weights, which gives smoother samples.

## Dataset

[**AFHQ** (Animal Faces-HQ)](https://huggingface.co/datasets/huggan/AFHQ): about 5,600 aligned cat faces, resized to 64×64. Downloaded automatically from Hugging Face (no login).

Also supported: [Oxford-IIIT Pets](https://huggingface.co/datasets/timm/oxford-iiit-pet) and your own folder of images (`<folder>/cat/*.jpg`, `<folder>/dog/*.jpg`).

## Results

<!-- Edit with your own observations after training -->
- **Vanilla GAN:** learns overall colours but produces blurry, noisy blobs. Dense layers have no notion of neighbouring pixels.
- **DCGAN:** produces recognisable (if blurry) cat faces with eyes, ears and fur patterns, a clear improvement from the convolutional architecture alone.
- **Latent interpolation** (below) changes smoothly between faces, which suggests the generator learned general features rather than memorising images.

![Latent interpolation](assets/dcgan_interpolation.png)

**Limitations:** 64×64 resolution, about 5.6k images and roughly 13k training steps on a free Colab GPU. Modern models (StyleGAN2, diffusion models) reach photo quality with far more data and compute.

## Run it

### 1. Train (Google Colab, free GPU, recommended)
Open `notebooks/gan_dcgan_colab.ipynb` in Colab, set **Runtime → Change runtime type → T4 GPU**, and choose **Run all**. It downloads AFHQ, trains both models, shows all results and downloads `gan_results.zip`.

Or train locally with:
```bash
pip install -r requirements.txt
python gan/train_gan.py --dataset afhq --class_name cat --model both --epochs 150 --batch_size 64
```

### 2. Generate images
Put `dcgan_generator.keras` (and optionally `gan_generator.keras`) in `model/`, then:
```bash
python generate.py --model dcgan --n 16 --seed 42      # saves generated.png
streamlit run app.py                                   # interactive generator in the browser
```

## Project structure
```
gan/gan_models.py              data loading, GAN + DCGAN models, training loop
gan/train_gan.py               command-line trainer
notebooks/gan_dcgan_colab.ipynb   Colab notebook: train, compare, visualise
generate.py                    make an image grid from a trained generator
app.py                         Streamlit generator demo
model/                         trained generators (*.keras) + info json
assets/                        result images used in this README
```

## References
- Goodfellow et al., *Generative Adversarial Networks*, 2014. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
- Radford, Metz & Chintala, *Unsupervised Representation Learning with Deep Convolutional GANs* (DCGAN), 2015. [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
- Choi et al., *StarGAN v2* (AFHQ dataset), 2020. [arXiv:1912.01865](https://arxiv.org/abs/1912.01865)
- Odena, Dumoulin & Olah, *Deconvolution and Checkerboard Artifacts*, Distill, 2016.

## Author
**Muzammil Khalid** · [github.com/mkdevelopss](https://github.com/mkdevelopss)

License: MIT
