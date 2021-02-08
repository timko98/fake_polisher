"""
Test implementation of the FakePolisher from https://arxiv.org/pdf/2006.07533v1.pdf
"""

import os
import pickle

import numpy as np
from PIL import Image

import imlib


def load_image(path, grayscale=False):
    """
    Load image.
    Copied from https://github.com/RUB-SysSec/GANDCTAnalysis
    """
    x = Image.open(path)
    # x = x.resize((64, 64))
    x = x.convert("L")
    x = np.asarray(x).ravel()

    return x


def main():
    """
    Load generated images and reconstruct them using a pca dictionary
    """
    data_folder = os.path.abspath("/path/to/data/folder")
    fake_folder = os.path.join(data_folder, "celeb_fake_spectral_128")
    pca_model = pickle.load(open(os.path.join(data_folder, "pca_dict/pca_model_50000_0.99_128.sav"), "rb"))

    D = pca_model.components_.T
    # Get representation vector

    for i, fn in enumerate(os.listdir(fake_folder)):
        y = load_image(os.path.join(data_folder, f"{fake_folder}/{fn}"))
        print(fn)
        # get representation vector x = ((D.T * D)^(-1)) * D.T * y
        x = np.matmul(np.matmul(np.linalg.inv(np.matmul(D.T, D)), D.T), y)
        # reconstruct y = D*x
        y_reconstruct = np.matmul(D, x).reshape(128, 128)
        # save image
        im = Image.fromarray(y_reconstruct)
        im = im.convert("L")
        im.save(os.path.join(data_folder, f"celeb_fake_reconstruct/{i}.jpg"))


if __name__ == '__main__':
    main()
