import numpy as np
import tensorflow as tf

from pystackreg import StackReg

from scipy.ndimage import shift
from skimage.transform import warp
from skimage.registration import phase_cross_correlation

from tensorflow.keras import layers

from matplotlib import pyplot as plt


def normalize(image, low=0, high=255, dtype=np.uint8):
    """This function is used to normalize the value range of the input image to [low, high].

    Args:
        :param image: the image which need to be normalized.
        :param low: the lower bound.
        :param high: the upper bound.
        :param dtype: output data type.
    """
    image = (image - image.min()) / (image.max() - image.min()) * (high - low) + low
    return image.astype(dtype)


def registration(img1, img2, method='phase_cross_correlation'):
    """This function is used to register 2 images.

    Args:
        :param img1: the first image.
        :param img2: the second image.
        :param method: which method is used for registration.

    List methods:
        A Pyramid Approach to Subpixel Registration Based on Intensity (http://bigwww.epfl.ch/thevenaz/stackreg/#Related)
        - translation
        - rigid (translation + rotation)
        - scaled rotation (translation + rotation + scaling)
        - affine (translation + rotation + scaling + shearing)
        - bilinear (non-linear transformation; does not preserve straight lines)
        - phase_cross_correlation
    """

    corrected_img2 = None
    if method == 'translation':
        sr = StackReg(StackReg.TRANSLATION)
        corrected_img2 = sr.register_transform(img1, img2)
    elif method == 'rigid':
        sr = StackReg(StackReg.RIGID_BODY)
        corrected_img2 = sr.register_transform(img1, img2)
    elif method == 'scaled_rotation':
        sr = StackReg(StackReg.SCALED_ROTATION)
        corrected_img2 = sr.register_transform(img1, img2)
    elif method == 'affine':
        sr = StackReg(StackReg.AFFINE)
        corrected_img2 = sr.register_transform(img1, img2)
    elif method == 'bilinear':
        sr = StackReg(StackReg.BILINEAR)
        corrected_img2 = sr.register_transform(img1, img2)
    elif method == 'phase_cross_correlation':
        shift_dis, _, _ = phase_cross_correlation(img1, img2)
        corrected_img2 = shift(img2, shift=shift_dis, mode='constant')

    return corrected_img2


def visualize_registration(img1, img2, corrected_img2):
    """This function is used to show 2 images sid-by-side to see the results of the registration.

    Args:
        :param img1: the first image.
        :param img2: the second image.
        :param img2: the corrected second image.
    """

    fig = plt.figure(figsize=(10, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(img1, cmap='gray')
    ax1.title.set_text('First image')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(img2, cmap='gray')
    ax2.title.set_text('Second image')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(corrected_img2, cmap='gray')
    ax3.title.set_text('Corrected Second image')

    plt.show()


def data_augmentation_rigid(image, rotation_range=0.01, height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05),
                            fill_mode='reflect', interpolation='bilinear', fill_value=0.0, flip_mode='horizontal_and_vertical'):
    """This function is used for data augmentation (rigid  transformation).

    Args:
        :param image: the original image.
        :param rotation_range: the range of the angle for the rotation.
        :param height_factor: the range of the vertical translation.
        :param width_factor: the range of the horizontal translation.
        :param fill_mode: the mode to decide which way to fill in the gap created by the translation.
        :param interpolation: the interpolation mode for the translation.
        :param fill_value: default value to fill in if fill_mode = constant
        :param flip_mode: flip mode including horizontal_and_vertical, horizontal, vertical
    """

    # define a sequence of functions for data augmentation
    augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip(flip_mode),  # flip
        layers.experimental.preprocessing.RandomTranslation(height_factor, width_factor, fill_mode=fill_mode, interpolation=interpolation, fill_value=fill_value),  # translation
        layers.experimental.preprocessing.RandomRotation(rotation_range),  # rotation
    ])

    return augmentation(image)


def data_augmentation_non_rigid(image, height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode='reflect', interpolation='bilinear', fill_value=0.0):
    """This function is used for data augmentation (non-rigid  deformation).

    Args:
        :param image: the original image.
        :param height_factor: the range of the vertical deformation.
        :param width_factor: the range of the horizontal deformation.
        :param fill_mode: the mode to decide which way to fill in the gap created by the deformation.
        :param interpolation: the interpolation mode for the deformation.
        :param fill_value: default value to fill in if fill_mode = constant
    """

    # define a sequence of functions for data augmentation
    augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomZoom(height_factor, width_factor=width_factor, fill_mode=fill_mode, interpolation=interpolation, fill_value=fill_value)  # deformation
    ])
    
    return augmentation(image)


def data_augmentation(image, use_rigid=False, use_non_rigid=True):
    """This function is used for data augmentation (non-rigid  deformation).

    Args:
        :param image: the original image.
        :param use_rigid: if you want to apply rigid transformation.
        :param use_non_rigid: if you want to apply non-rigid deformation.
    """

    augmented_image = tf.expand_dims(image, 0)  # reshape from H x W x C to N x H x W x C

    if use_rigid:  # in case of rigid transformation
        augmented_image = data_augmentation_rigid(augmented_image)

    if use_non_rigid:  # in case of non-rigid deformation
        augmented_image = data_augmentation_non_rigid(augmented_image)

    return augmented_image


def standardize(image, image_size=224):
    """This function is used for data augmentation (non-rigid  deformation).

    Args:
        :param image: the original image (range [0, 255])
        :param image_size: expected input image size of the network.
    """

    if image.shape[2] == 1:
        image = np.concatenate((image, image, image), axis=2)

    # define a sequence of functions for data standardization
    resize_and_rescale = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(image_size, image_size),  # resize to match the input size of the network
        layers.experimental.preprocessing.Rescaling(1.0 / 255.0)  # normalize to 0->1
    ])

    return resize_and_rescale(image)
