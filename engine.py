import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from utils import normalize, standardize, data_augmentation
from metrics import precision_m, recall_m, similarity_loss
from models import create_network


class Dataset(object):
    def __init__(self, image1_path, image2_path, label_path, n_classes=2, training=True, use_rigid=False, use_non_rigid=True):
        """Init the dataset handler

        :param image1_path: list of the first type of images. (ADC)
        :param image2_path: list of the second type of images. (T2WI)
        :param label_path: list of the label
        :param use_rigid: if you want to apply rigid transformation in data augmentation.
        :param use_non_rigid: if you want to apply non-rigid deformation in data augmentation.
        """
        # NOTE: you have to modify the code to read your data, the below variables are the ones you need to modify
        # image1_path, image2_path, label_path

        # there is two ways, one is to load whole data to the memory
        # one is to load file-by-file. In this case, I load all the data to memory.
        # You can modify the path as you wish later. In this case, I just load the mnist dataset.
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()

        # as mentioned earlier, these variables are used for the data augmentation
        self.use_rigid = use_rigid
        self.use_non_rigid = use_non_rigid

        # flag to indicate if the is used for training.
        self.training = training

        # just number of classes
        self.n_classes = n_classes

    def read_data(self, index):
        """This function is used to read the data with the index

        :param index: the index of the data you want to get.
        """

        # if this is for training, just load the the from training list
        if self.training:
            x1 = self.train_images[index]  # the first list of images (ADC)
            x2 = self.train_images[index]  # the second list of images (T2WI)
            y = self.train_labels[index]   # the list of labels
        else:  # if this is for testing, just load the the from testing list
            x1 = self.test_images[index]  # the first list of images (ADC)
            x2 = self.test_images[index]  # the second list of images (T2WI)
            y = self.test_labels[index]   # the list of labels
        
        height, width = x1.shape  # get the size of the image
        x1 = normalize(x1.reshape(height, width, 1))  # apply the normalization (norm to range [0, 1])
        x1 = standardize(x1)                          # apply the standardization (reshape the data)

        x2 = normalize(x2.reshape(height, width, 1))  # apply the normalization (norm to range [0, 1])
        x2 = standardize(x2)                          # apply the standardization (reshape the data)

        # apply data augmentation
        augmented_data = data_augmentation(np.concatenate([x1, x2], axis=2), use_rigid=self.use_rigid, use_non_rigid=self.use_non_rigid)

        return augmented_data[:, :, :, :3], augmented_data[:, :, :, 3:], keras.utils.to_categorical(y, num_classes=self.n_classes)

    def len(self):
        """This function is used to get the size of the dataset (number of samples)
        """
        # if this is for training, get the size of the training set
        if self.training:
            return self.train_images.shape[0]
        else:  # if this is for testing, get the size of the testing set
            return self.test_images.shape[0]


class DataLoader(keras.utils.Sequence):
    def __init__(self, image1_path, image2_path, label_path, fusion_mode, batch_size=32, dim=(224, 224), n_channels=3, n_classes=2, shuffle=True, training=True, use_rigid=False, use_non_rigid=True):
        """Init the dataset loader

        :param image1_path: list of the first type of images. (ADC)
        :param image2_path: list of the second type of images. (T2WI)
        :param label_path: list of the label
        :param use_rigid: if you want to apply rigid transformation in data augmentation.
        :param use_non_rigid: if you want to apply non-rigid deformation in data augmentation.
        """

        super().__init__()  # just call the initialization of the parent class

        # NOTE: you have to modify the code to read your data, the below variables are the ones you need to modify
        # image1_path, image2_path, label_path
        self.dataset = Dataset(image1_path, image2_path, label_path, n_classes=n_classes, use_rigid=self.use_rigid, use_non_rigid=self.use_non_rigid)  # init the data handler

        self.list_indices = np.arange(self.dataset.len())  # get list of indices based on the size of the data
        self.batch_size = batch_size  # just the batch size (number of samples you want to process at the same time)
        self.dim = dim                # the resolution of the images
        self.n_channels = n_channels  # number of channels of the input images
        self.n_classes = n_classes    # number of classes of your tasks
        self.shuffle = shuffle        # flag if we need to shuffle the data
        
        self.fusion_mode = fusion_mode  # fusion mode that are used in the paper
        self.training = training        # flag to indicate if the is used for training.

        self.on_epoch_end()             # shuffle the data if needed

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        # just shuffle the data if needed
        self.indexes = np.arange(len(self.list_indices))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.list_indices) / self.batch_size))

    def __getitem__(self, index):
        """Obtain the data with index

        :param index: the index of the data you want to get.
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        
        # Find list of indices
        list_indices = [self.list_indices[k] for k in indexes]
        
        # Init the holders for the data
        X1 = np.empty((self.batch_size, *self.dim, self.n_channels))
        X2 = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Read the data
        for i, idx in enumerate(list_indices):
            X1[i, ], X2[i, ], Y[i, ] = self.dataset.read_data(idx)

        # if the fusion mode is the third strategy in the paper, we need to have one more dump label to bypass the
        # template of the keras platform. Z will not be use to calculate anything
        if self.fusion_mode == 'paper':
            Z = np.empty((self.batch_size, 1), dtype=int)
            return [X1, X2], (Y, Y, Y, Z)
        else:  # return 3 lists of labels (one for the adc, one for the t2wi, one for the fusion)
            return [X1, X2], (Y, Y, Y)


def train(modality='resnet50', combine_mode='c1', fusion_mode='paper', saved_name='pretrained'):
    """This function is used to train the CNN for the deep network.

    Args:
        :param modality: the modality to use (resnet50, vgg16, googlenet).
        :param fusion_mode: the fusion mode to use (concatenate, sum, paper, adc, t2wi) (using concatenation, element-wise sum, and the proposed one in the paper, only use adc, only use t2wi).
        :param combine_mode: c1, c2, c3. c1 just concat, c2 pca then concat, c3 concat prediction.
        :param saved_name: the name to save the model.
    """
    n_classes = 10  # number of the classes
    batch_size = 2  # batch size
    n_epochs = 500  # the number of epochs

    # make an instance of data loader
    training_generator = DataLoader(None, None, None, fusion_mode=fusion_mode, n_classes=n_classes, batch_size=batch_size)

    # create the network
    model = create_network(modality=modality, combine_mode=combine_mode, fusion_mode=fusion_mode, image_size=224, image_dim=3, n_classes=n_classes, training=True)

    # if the fusion strategy is to use concatenation or element-wise sum, or just using only adc or t2wi
    # the only loss function we need is categorical_crossentropy, 
    if fusion_mode in ['sum', 'concatenate', 'adc', 't2wi']:
        model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy())
    else:  # if it is the third strategy in the paper, we also need the similarity loss
        # define list of lost
        losses = {
            "classification_1": "categorical_crossentropy",
            "classification_2": "categorical_crossentropy",
            "classification": "categorical_crossentropy",
            "tf.math.reduce_mean_1": similarity_loss,
        }
        # define list of weights for the lost
        loss_weights = {"classification_1": 1.0, "classification_2": 1.0, "classification": 1.0, "tf.math.reduce_mean_1": 1.0}

        # compille the model (keras thing)
        model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights)

    # Train model on dataset
    model.fit(x=training_generator, epochs=n_epochs, verbose=True)  # optimize the parameters
    model.save_weights(saved_name + '_' + modality + '.h5')         # save at the end of the training


def get_pretrained_model(modality='resnet50', combine_mode='c1', fusion_mode='paper', saved_name='pretrained'):
    """This function is used to get the pretrained model the CNN.

    Args:
        :param modality: the modality to use (resnet50, vgg16, googlenet).
        :param fusion_mode: the fusion mode to use (concatenate, sum, paper, adc, t2wi) (using concatenation, element-wise sum, and the proposed one in the paper, only use adc, only use t2wi).
        :param combine_mode: c1, c2, c3. c1 just concat, c2 pca then concat, c3 concat prediction.
        :param saved_name: the name to save the model.
    """
    n_classes = 10
    # batch_size = 2

    model = create_network(modality=modality, combine_mode=combine_mode, fusion_mode=fusion_mode, image_size=224, image_dim=3, n_classes=n_classes, training=False)
    model.load_weights(saved_name + '_' + modality + '.h5')
    model.compile(metrics=['accuracy', precision_m, recall_m])

    # Train model on dataset
    # testing_generator = DataLoader(None, None, None, fusion_mode=fusion_mode, n_classes=n_classes, batch_size=batch_size, shuffle=False)
    # model.evaluate(x=testing_generator, verbose=True)

    return model
