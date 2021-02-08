"""
Adapted from https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Pytorch
Script to generate image samples from trained GAN models.
"""

import os
import sys

import numpy as np
import torch
import tqdm

import torchlib
import imlib

from src import module, module_spectrum

ROOT_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")

device = torch.device("cuda")

G_normal = module.ConvGenerator(128, 3, n_upsamplings=4).to(device)
G_spectral = module_spectrum.ConvGenerator(128, 3, n_upsamplings=5).to(device)


def load_model(path, model):
    for file in os.listdir(path):
        if model in file:
            print(f"Loading the following model for sampling: {file}")

            try:
                ckpt = torchlib.load_checkpoint(os.path.join(path, file))

                if model == "(1)":
                    G_normal.load_state_dict(ckpt['G'])
                    return G_normal
                elif model == "spectral":
                    G_spectral.load_state_dict(ckpt['G'])
                    return G_spectral

            except Exception as e:
                print(str(e))

    return None


def sample(model, image_real):
    with torch.no_grad():
        return model(image_real)


def main():
    model_path = os.path.join(ROOT_PATH, f"data{os.path.sep}GAN_models{os.path.sep}")
    model = load_model(model_path, "(1)")
    if model is None:
        sys.exit("No model could be loaded.")
    else:
        print("Successfully loaded the GAN model.")

    out_dir = os.path.join(ROOT_PATH, f"data{os.path.sep}celeb_fake")

    model.eval()
    for i in tqdm.tqdm(range(1000)):
        torch.manual_seed(i)
        sample_noise = torch.randn(1, 128, 1, 1).to(device)
        image_fake = np.transpose(sample(model, sample_noise).data.cpu().numpy(), (0, 2, 3, 1)).squeeze()
        # image_fake = imlib.imresize(image_fake, (64,64))
        imlib.imwrite(image_fake, f"{out_dir}{os.path.sep}img_{i}.jpg")


if __name__ == '__main__':
    main()
