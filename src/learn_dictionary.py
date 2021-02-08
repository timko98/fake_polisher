"""
Learn a pca dictionary. Methodology inspired by https://arxiv.org/pdf/2006.07533v1.pdf
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

PCA_COMPONENTS = 0.9


def _pca(array):
    """
    pca helper function
    """
    pca = PCA(n_components=PCA_COMPONENTS)
    return pca.fit(array)


def learn_model(real_folder, method="pca"):
    """
    Learn a pca dictionary from real data
    """
    real_images = [os.path.join(real_folder, path) for path in sorted(os.listdir(real_folder))]
    images = np.array([load_image(x) for x in real_images[:20000]])
    if method == "pca":
        pca_model = _pca(images)
        return pca_model


def load_image(path, grayscale=False):
    """
    Load image.
    Copied from https://github.com/RUB-SysSec/GANDCTAnalysis
    """
    x = Image.open(path)
    # x = x.resize((64,64))
    x = x.convert("L")
    x = np.asarray(x).ravel()

    return x


def main():
    data_folder = os.path.abspath("/path/to/data/folder")
    real_folder = os.path.join(data_folder, "celeb_real")
    model = learn_model(real_folder, "pca")
    pickle.dump(model, open("../datasets/pca_dict/pca_model_20000_0.9_128.sav", "wb"))
    fig, axes = plt.subplots(2, 9, figsize=(9, 3),
                             subplot_kw={"xticks": [], "yticks": []},
                             gridspec_kw=dict(hspace=0.01, wspace=0.01))
    print(model.components_.shape)
    for i, ax in enumerate(axes.flat):
        ax.imshow(model.components_[i].reshape(128, 128), cmap="gray")

    plt.show()


if __name__ == '__main__':
    main()
