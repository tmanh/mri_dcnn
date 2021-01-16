# Automated diagnosis of prostate cancer in multi-parametric MRI based on multimodal convolutionalneural networks

This is the implementation of the paper: Automated diagnosis of prostate cancer in multi-parametric MRI based on multimodal convolutionalneural networks, MH Le, 2017.

> conda create -n mri python=3.7
> conda activate mri
> conda install numpy
> conda install -c conda-forge nibabel
> conda install scikit-image
> pip install pystackreg
> pip install tensorflow-gpu (or tensorflow)
> conda install -c conda-forge scikit-learn
> pip install mahotas

## Preprocessing

The processing of the paper "Automated diagnosis of prostate cancer in multi-parametric MRI based on multimodal convolutionalneural networks" requires the image registration library called MIRT. I have write a sample [script](./MIRT/preprocess.m) to register all the data. You have to modify or write a simple conversion data code from your npy data to png image to use the code.
## Data augmentation

The data augmentation code is in [utils file](./utils.py) including both non-rigid (data_augmentation_non_rigid) and rigid (data_augmentation_rigid) appoaches. The code is implemented with Tensorflow (v2.4).

# Network implementation
