"""
Plotting of the frequency spectra for the real and fake images. Inspired by https://github.com/RUB-SysSec/GANDCTAnalysis
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.filters import window

from src.math_utils import welford


def plot_magnitude(mean_r, mean_f):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, dpi=300)
    fig.suptitle("Magnitude spectrum", fontsize=16)

    mat1 = ax1.matshow(mean_r, cmap="gray")
    mat2 = ax2.matshow(mean_f, cmap="gray")

    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax1.set_title("Real")
    ax2.set_title("Fake")

    _1 = fig.colorbar(mat1, ax=ax1, fraction=0.046, pad=0.04)
    _2 = fig.colorbar(mat2, ax=ax2, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.show()


def _plot(outpath, metric_r, metric_f, metric_name):
    """
    Plotting funktionality.
    """
    max_value_r = np.asarray(list(map(lambda x: x[1].max(), metric_r))).max()
    max_value_f = np.asarray(list(map(lambda x: x[1].max(), metric_f))).max()
    min_value_r = np.asarray(list(map(lambda x: x[0].max(), metric_r))).min()
    min_value_f = np.asarray(list(map(lambda x: x[0].max(), metric_f))).min()

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, dpi=300)
    fig.suptitle(metric_name, fontsize=16)

    # Possible colormap styles: Spectral, spring, jet, coolwarm, inferno
    mat1 = ax1.matshow(metric_r, cmap=plt.cm.jet, vmax=max_value_r, vmin=min_value_r)
    mat2 = ax2.matshow(metric_f, cmap=plt.cm.jet, vmax=max_value_f, vmin=min_value_f)

    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax1.set_title("Real")
    ax2.set_title("Fake")

    _1 = fig.colorbar(mat1, ax=ax1, fraction=0.046, pad=0.04)
    _2 = fig.colorbar(mat2, ax=ax2, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.show()


def calculate_mean_std(real_folder, fake_folder):
    """
    Calculate the mean and the std of the frequency spectra.
    Adapted from https://github.com/RUB-SysSec/GANDCTAnalysis
    """
    # statistics for real images
    real_images = [os.path.join(real_folder, path) for path in sorted(os.listdir(real_folder))]
    images_r = map(lambda d: load_image(
        d, grayscale=True), real_images[:1000])
    images_r_dct = map(dct2, images_r)
    mean_r, variance_r = welford(images_r_dct)

    # statistics for fake images
    fake_images = [os.path.join(fake_folder, path) for path in sorted(os.listdir(fake_folder))]
    images_f = map(lambda d: load_image(
        d, grayscale=True), fake_images)
    images_f_dct = map(dct2, images_f)
    mean_f, variance_f = welford(images_f_dct)
    return 20 * log_scale(mean_r), 20 * log_scale(mean_f)


def dct2(array):
    """
    Calculate 2D FFT for array.
    """
    array = array * window('hann', array.shape)
    f = np.fft.fft2(array)
    fshift = np.fft.fftshift(f)
    return fshift


def log_scale(array, epsilon=1e-12):
    """
    Log scale the input array.
    Copied from https://github.com/RUB-SysSec/GANDCTAnalysis
    """
    array = np.abs(array)
    array += epsilon  # no zero in log
    array = np.log(array)
    return array


def load_image(path, grayscale=False, tf=False):
    """
    Load image.
    Copied from https://github.com/RUB-SysSec/GANDCTAnalysis
    """
    x = Image.open(path)

    if grayscale:
        x = x.convert("L")
        if tf:
            x = np.asarray(x)
            x = np.reshape(x, [*x.shape, 1])

    return np.asarray(x)


def main():
    data_folder = os.path.abspath("path/to/data/folder")
    real_folder = os.path.join(data_folder, "celeb_fake_normal")
    fake_folder = os.path.join(data_folder, "celeb_fake_spectral")
    magnitude_r, magnitude_f = calculate_mean_std(real_folder, fake_folder)
    plot_magnitude(magnitude_r, magnitude_f)


if __name__ == '__main__':
    main()
