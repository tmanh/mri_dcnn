import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from utils import normalize, standardize, data_augmentation


class Dataset(object):
    def __init__(self, image1_path, image2_path, label_path, training=True):
        # there is two ways, one is to load whole data to the memory
        # one is to load file-by-file. In this case, I load all the data to memory.
        # You can modify the path as you wish later. In this case, I just load the mnist dataset.
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()

        self.training = training

    def read_data(self, index):
        if self.training:
            x1 = self.train_images[index]
            x2 = self.train_images[index]
            y = self.train_labels[index]
        else:
            x1 = self.test_images[index]
            x2 = self.test_images[index]
            y = self.test_labels[index]

        x1 = normalize(x1)
        x1 = standardize(x1)

        x2 = normalize(x2)
        x2 = standardize(x2)

        augmented_data = data_augmentation(np.concatenate([x1, x2]), axis=2)

        return augmented_data[:, :, :, :3], augmented_data[:, :, :, 3:], y

    def len(self):
        if self.training:
            return len(self.train_images)
        else:
            return len(self.test_images)

class DataLoader(keras.utils.Sequence):
    def __init__(self, image1_path, image2_path, label_path, batch_size=32, dim=(224, 224), n_channels=3, n_classes=10, shuffle=True, training=True):
        super().__init__()

        self.dataset = Dataset()

        self.list_indices = np.arange(self.dataset.len())
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle

        self.training = training

        self.on_epoch_end()

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_indices))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.list_indices) / self.batch_size))

    def __getitem__(self, index):
        """Obtain the data with index
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        
        # Find list of indices
        list_indices = [self.list_indices[k] for k in indexes]
        
        # Generate data
        X1 = np.empty((self.batch_size, *self.dim, self.n_channels))
        X2 = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, 1), dtype=int)

        for i in list_indices:
            X1[i, ], X2[i, ], Y[i, ] = self.dataset.read_data(i)
        
        return X1, X2, Y


def train(modality='vgg16'):
    training_generator = DataLoader(None, None, None)
    validation_generator = DataLoader(None, None, None)
