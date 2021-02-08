# Fake Polisher Test Implementation

This repository contains the code that was used for the seminar "Seminar Adversarial Machine Learning 2020/2021"
at Karlsruhe Institute of Technology.  

Some code was copied or adapted from the follwoing sources (referenced in the code):

* Paper about Spectral Regularization: https://github.com/cc-hpc-itwm/UpConv
* Pytorch impl. of DCGAN, LSGAN, WGAN-GP(LP) and DRAGAN: https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Pytorch
* Paper - Leveraging Frequency Analysis for Deep Fake Image Recognition: https://github.com/RUB-SysSec/GANDCTAnalysis

Some code was newly implemented:

* FakePolisher: https://arxiv.org/pdf/2006.07533v1.pdf


---

Prerequisites to reproduce the results:

1. Install the required libraries.
    ```python 
       pip install -r requirements.txt
    ```
2. Acquire a training dataset (in this case, we used [celebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html))
3. Train GAN models using the mentioned papers above (with or without spectral regularization) or use pretrained models
4. (Optional) Reconstruct images using the fake polisher module.
5. Run a classifier on either raw pixels of images or on images transformed into the frequency representation.

--- 

src - scripts:

- **prepare_datasets.py**:
    Script that loads the images from the datasets, transforms them into the frequency representation (if enabled) and creates and train-test split for the classifier.  
    The path to the datasets folder has to be set.

- **classification.py**:
    A simple script that trains a classifier and tests it on either raw pixels or transformed images.
    
- **learn_dictionary**:
    Script to learn a dictionary model of real images by using PCA.
    The path to the datasets folder has to be set.
    
- **fake_polish.py**:
    Script to reconstruct GAN-generated images using the learned PCA-dictionary.  
    The path to the datasets folder has to be set. 
    
- **data.py**, **module.py**, **sample.py**, **spectrum.py**, **module_spectrum.py**, **math_utils.py**:
    Utils and helper functions copied and adapted from the above mentioned existing implementations.
