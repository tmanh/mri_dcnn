
import numpy as np
import nibabel as nib

from utils import registration, visualize_registration, data_augmentation, standardize, normalize
from matplotlib import pyplot as plt

# load the sample data
t1_img = nib.load('./data/mni_icbm152_t1_tal_nlin_asym_09a.nii')
t1_data = np.asanyarray(t1_img.dataobj)
t2_img = nib.load('./data/mni_icbm152_t2_tal_nlin_asym_09a.nii')
t2_data = np.asanyarray(t2_img.dataobj)

t1_slice = t1_data[:, :, 94]
t2_slice = t2_data[:, :, 94]

t2_slice[15:, :] = t2_slice[:-15, :]

height, width = t1_slice.shape

# data augmentation
plt.figure(figsize=(10, 10))
for i in range(9):
    # reshape to H x W x C
    tmp = t1_slice.reshape(height, width, 1)

    # normalize the data to [0, 255]
    tmp = normalize(tmp)

    # standardize the input data to match input dimension of the CNN
    tmp = standardize(tmp)

    # data augmentation
    augmented_image = data_augmentation(tmp)  # N x H x W x C

    # plot the image
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_image[0])
    plt.axis("off")

plt.show()
