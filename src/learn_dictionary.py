import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

PCA_COMPONENTS = 0.99


def _pca(array):
    pca = PCA(n_components=PCA_COMPONENTS)
    return pca.fit(array)


def learn_model(real_folder, method="pca"):
    real_images = [os.path.join(real_folder, path) for path in sorted(os.listdir(real_folder))]
    images = np.array([load_image(x) for x in real_images[:50000]])
    if method == "pca":
        pca_model = _pca(images)
        return pca_model


def load_image(path, grayscale=False):
    """
    Load image.
    Copied from https://github.com/RUB-SysSec/GANDCTAnalysis
    """
    x = Image.open(path)

    x = x.convert("L")
    x = np.asarray(x).ravel()

    return x


def main():
    data_folder = os.path.abspath("C:\\Users\\Tim\\git-repos\\fake_polisher\\data")
    real_folder = os.path.join(data_folder, "test\\real")
    model = learn_model(real_folder, "pca")
    pickle.dump(model, open("pca_model_50000_0.99.sav", "wb"))
    fig, axes = plt.subplots(2, 9, figsize=(9, 3),
                             subplot_kw={"xticks": [], "yticks": []},
                             gridspec_kw=dict(hspace=0.01, wspace=0.01))
    print(model.components_.shape)
    for i, ax in enumerate(axes.flat):
        ax.imshow(model.components_[i].reshape(256, 256), cmap="gray")

    plt.show()


if __name__ == '__main__':
    main()
