import os

from PIL import Image
import numpy as np
from scipy import fftpack
from sklearn.model_selection import StratifiedShuffleSplit


def load_image(path):
    """
    Load image.
    Copied from https://github.com/RUB-SysSec/GANDCTAnalysis
    """
    x = Image.open(path)
    # x = x.resize((64, 64))
    x = x.convert("L")
    x = np.asarray(x).ravel()

    return x


def dct2(array):
    """
    Calculate 2D DCT for array.
    Copied from https://github.com/RUB-SysSec/GANDCTAnalysis
    """
    array = array.reshape((128,128))
    array = fftpack.dct(array, type=2, norm="ortho", axis=0)
    array = fftpack.dct(array, type=2, norm="ortho", axis=1)
    return array.ravel()


def get_data_labels(dct):
    data_folder = os.path.abspath("/path/to/data")
    fake_folder = os.path.join(data_folder, "celeb_fake_spectral_128")
    real_folder = os.path.join(data_folder, "celeb_real")

    fake_images = [os.path.join(fake_folder, path) for path in sorted(os.listdir(fake_folder))]
    real_images = [os.path.join(real_folder, path) for path in sorted(os.listdir(real_folder))]

    images_f = [load_image(d) for d in fake_images]
    images_r = [load_image(d) for d in real_images[:20000]]

    if dct:
        images_f = [dct2(d) for d in images_f]
        images_r = [dct2(d) for d in images_r]

    data = np.vstack((np.array(images_f), np.array(images_r)))
    # fake image labels are 1, real image labels are 0
    labels = np.hstack((np.ones(len(images_f)), np.zeros(len(images_r))))

    return data, labels


def create_test_train_split(data, labels):
    """
    Function to create balanced train and test data sets such that the label distributions are equal.
    :param data: 2D numpy array of the data to be split.
    :param labels: 1D numpy array of the labels.
    :param test_size: percent of the data that should be used for testing (e.g. 0.2, 0.33, ..)
    :return: train and test sets for the data (2D) and the labels (1D) as numpy arrays
    """
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

    for train_idx, test_idx in strat_split.split(data, labels):
        X_train, X_test = data[train_idx], data[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        print(f"Percentage of label 1 in training set: {np.count_nonzero(y_train) / (y_train.shape[0])}")
        print(f"Percentage of label 1 in test set: {np.count_nonzero(y_test) / (y_test.shape[0])}")

    return X_train, X_test, y_train, y_test


def prepare_datasets(dct):
    data, labels = get_data_labels(dct)
    X_train, y_train, X_test, y_test = create_test_train_split(data, labels)
    return X_train, y_train, X_test, y_test
