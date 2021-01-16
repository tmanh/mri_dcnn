import pickle

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.decomposition import PCA

from utils import normalize, standardize, data_augmentation
from metrics import precision_m, recall_m, similarity_loss, precision, recall
from models import create_network
from handcraft import compute_handcraft_features, compute_handcraft_features, compute_haralick_features


class Dataset(object):
    def __init__(self, image1_path, image2_path, label_path, training=True, use_rigid=False, use_non_rigid=True):
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
        self.train_images = self.train_images[:100, :, :]
        self.train_labels = self.train_labels[:100]
        self.test_images = self.test_images[:100, :, :]
        self.test_labels = self.test_labels[:100]

        # as mentioned earlier, these variables are used for the data augmentation
        self.use_rigid = use_rigid
        self.use_non_rigid = use_non_rigid

        # flag to indicate if the is used for training.
        self.training = training

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

        # NOTE: because the data I used has multiple classes, so I have to modified it a bit, use the commented code below instead
        # return augmented_data[:, :, :, :3], augmented_data[:, :, :, 3:], y

        # NOTE: remove the following lines, use the above in your case
        y = (y == 2).astype(np.uint8)
        return augmented_data[:, :, :, :3], augmented_data[:, :, :, 3:], y

    def len(self):
        """This function is used to get the size of the dataset (number of samples)
        """
        # if this is for training, get the size of the training set
        if self.training:
            return self.train_images.shape[0]
        else:  # if this is for testing, get the size of the testing set
            return self.test_images.shape[0]


class DataLoader(keras.utils.Sequence):
    def __init__(self, image1_path, image2_path, label_path, fusion_mode, batch_size=32, dim=(224, 224), n_channels=3, shuffle=True, training=True, use_rigid=False, use_non_rigid=True):
        """Init the dataset loader

        :param image1_path: list of the first type of images. (ADC)
        :param image2_path: list of the second type of images. (T2WI)
        :param fusion_mode: the fusion mode to use (concatenate, sum, paper, adc, t2wi) (using concatenation, element-wise sum, and the proposed one in the paper, only use adc, only use t2wi).
        :param label_path: list of the label
        :param use_rigid: if you want to apply rigid transformation in data augmentation.
        :param use_non_rigid: if you want to apply non-rigid deformation in data augmentation.
        """

        super().__init__()  # just call the initialization of the parent class

        # NOTE: you have to modify the code to read your data, the below variables are the ones you need to modify
        # image1_path, image2_path, label_path
        self.dataset = Dataset(image1_path, image2_path, label_path, use_rigid=use_rigid, use_non_rigid=use_non_rigid)  # init the data handler

        self.list_indices = np.arange(self.dataset.len())  # get list of indices based on the size of the data
        self.batch_size = batch_size  # just the batch size (number of samples you want to process at the same time)
        self.dim = dim                # the resolution of the images
        self.n_channels = n_channels  # number of channels of the input images
        self.shuffle = shuffle        # flag if we need to shuffle the data
        
        self.fusion_mode = fusion_mode  # fusion mode that are used in the paper
        self.training = training        # flag to indicate if the is used for training.

        # as mentioned earlier, these variables are used for the data augmentation
        self.use_rigid = use_rigid
        self.use_non_rigid = use_non_rigid

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
        Y = np.empty((self.batch_size, 2), dtype=int)

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
    n_classes = 1  # number of the classes
    batch_size = 2  # batch size
    n_epochs = 500  # the number of epochs

    # make an instance of data loader
    # NOTE: fill in the paths to your data
    training_generator = DataLoader(None, None, None, fusion_mode=fusion_mode, batch_size=batch_size)

    # create the network
    model = create_network(modality=modality, combine_mode=combine_mode, fusion_mode=fusion_mode, image_size=224, image_dim=3, n_classes=n_classes, training=True)

    # if the fusion strategy is to use concatenation or element-wise sum, or just using only adc or t2wi
    # the only loss function we need is categorical_crossentropy, 
    if fusion_mode in ['sum', 'concatenate', 'adc', 't2wi']:
        model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy())
    else:  # if it is the third strategy in the paper, we also need the similarity loss
        # define list of lost
        losses = {
            "tf.math.sigmoid": "categorical_crossentropy",
            "tf.math.sigmoid_1": "categorical_crossentropy",
            "tf.__operators__.add": "categorical_crossentropy",
            "tf.math.reduce_mean_1": similarity_loss,
        }

        # define list of weights for the lost
        loss_weights = {"tf.math.sigmoid": 1.0, "tf.math.sigmoid_1": 1.0, "tf.__operators__.add": 1.0, "tf.math.reduce_mean_1": 1.0}

        # compille the model (keras thing)
        model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights)

    # Train model on dataset
    model.fit(x=training_generator, epochs=n_epochs, verbose=True)  # optimize the parameters
    model.save_weights(saved_name + '_' + modality + '.h5')         # save at the end of the training


def predict(image1_path, image2_path, label_path, modality='resnet50', combine_mode='c1', fusion_mode='paper', saved_model_name='pretrained', mode='train'):
    """This function is used to obtain all the predictions and merged functions. This function only work on resnet50.

    Args:
        :param image1_path: list of the first type of images. (ADC)
        :param image2_path: list of the second type of images. (T2WI)
        :param label_path: list of the label
        :param modality: the modality to use (resnet50, vgg16, googlenet).
        :param fusion_mode: the fusion mode to use (concatenate, sum, paper, adc, t2wi) (using concatenation, element-wise sum, and the proposed one in the paper, only use adc, only use t2wi).
        :param combine_mode: c1, c2, c3. c1 just concat, c2 pca then concat, c3 concat prediction.
        :param saved_model_name: the name to save the model.
        :param mode: indicate if we run on train data or test data.
    """

    n_classes = 1
    batch_size = 1

    if mode == 'train':
        model = create_network(modality=modality, combine_mode=combine_mode, fusion_mode=fusion_mode, image_size=224, image_dim=3, n_classes=n_classes, training=True)
    else:
        model = create_network(modality=modality, combine_mode=combine_mode, fusion_mode=fusion_mode, image_size=224, image_dim=3, n_classes=n_classes, training=False)
    model.load_weights(saved_model_name + '_' + modality + '.h5')
    model.compile()

    # Train model on dataset
    testing_generator = DataLoader(image1_path, image2_path, label_path, fusion_mode=fusion_mode, batch_size=batch_size, shuffle=False)

    predictions = []
    merged_features = []
    for i in range(len(testing_generator)):
        input_data, _ = testing_generator[i]
        _, _, predict, merge_features = model.predict(x=input_data, verbose=True)
        
        predictions.append(predict)
        merged_features.append(merge_features)

    predictions = np.array(predictions)
    merged_features = np.array(merged_features)

    f1 = open(modality + '_' + mode + '_prediction' + '.pkl', 'wb')
    pickle.dump(predictions, f1)
    f1.close()

    f2 = open(modality + '_' + mode + '_merged_features' + '.pkl', 'wb')
    pickle.dump(merged_features, f2)
    f2.close()


def test(modality='resnet50', combine_mode='c1', fusion_mode='paper', saved_name='pretrained'):
    """This function is used to test model the CNN.

    Args:
        :param modality: the modality to use (resnet50, vgg16, googlenet).
        :param fusion_mode: the fusion mode to use (concatenate, sum, paper, adc, t2wi) (using concatenation, element-wise sum, and the proposed one in the paper, only use adc, only use t2wi).
        :param combine_mode: c1, c2, c3. c1 just concat, c2 pca then concat, c3 concat prediction.
        :param saved_name: the name to save the model.
    """
    n_classes = 1
    batch_size = 2

    model = create_network(modality=modality, combine_mode=combine_mode, fusion_mode=fusion_mode, image_size=224, image_dim=3, n_classes=n_classes, training=False)
    model.load_weights(saved_name + '_' + modality + '.h5')
    model.compile(metrics=['accuracy', precision_m, recall_m])

    # Train model on dataset
    testing_generator = DataLoader(None, None, None, fusion_mode=fusion_mode, batch_size=batch_size, shuffle=False)
    model.evaluate(x=testing_generator, verbose=True)

    return model


def svm_stage_1(saved_name='svm'):
    """This function is used to train and get the output results of the SVM.

    Args:
        :param saved_name: the name to save the model.
    """

    # load all the data to memory.
    # NOTE: you have to modify this to read the data
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images[:100, :, :]
    train_labels = (train_labels[:100] == 2).astype(np.uint8)
    test_images = test_images[:100, :, :]
    test_labels = (test_labels[:100] == 2).astype(np.uint8)

    # NOTE: I duplicate the process because this data only has one image input
    # so you have to modify the code to compute the features from each input image
    # then, concat them
    train_features = []
    for i in range(train_images.shape[0]):
        first_order_features = compute_handcraft_features(train_images[i])
        second_order_features = compute_haralick_features(train_images[i])
        
        features = np.concatenate([first_order_features, second_order_features, first_order_features, second_order_features])
        train_features.append(features)

    test_features = []
    for i in range(test_images.shape[0]):
        first_order_features = compute_handcraft_features(test_images[i])
        second_order_features = compute_haralick_features(test_images[i])

        features = np.concatenate([first_order_features, second_order_features, first_order_features, second_order_features])
        test_features.append(features)

    # optimize by SVM
    print('Optimization started.')
    estimator = SVR(kernel="linear", max_iter=200000)  # init svm
    selector = RFE(estimator, n_features_to_select=100, step=2)  # init rfe
    selector = selector.fit(train_features, train_labels)  # optimize
    print('Optimization ended.')
    train_predicted = selector.predict(train_features)  # predict the output on test set
    test_predicted = selector.predict(test_features)  # predict the output on test set

    precision_results = precision(test_labels, test_predicted)
    recall_results = recall(test_labels, test_predicted)

    print('Precision: {} - Recall: {}'.format(precision_results, recall_results))

    f = open(saved_name + '.pkl', 'wb')
    pickle.dump(selector, f)
    f.close()

    f = open(saved_name + '_test_prediction.pkl', 'wb')
    pickle.dump(test_predicted, f)
    f.close()

    f = open(saved_name + '_train_prediction.pkl', 'wb')
    pickle.dump(train_predicted, f)
    f.close()


def svm_stage_2_combine_1(cnn_train_merged_features='resnet50_train_merged_features.pkl', cnn_test_merged_features='resnet50_test_merged_features.pkl'):
    """This function is used to train and get the output results of the SVM (stage 2).
    For this strategy, the paper suggested to concat the features of the CNN and the features of stage-1 SVM instead of using the results of stage-1 SVM.

    Args:
        :param saved_cnn_name: the name to save the model.
    """
    print('Read CNN features')

    # load cnn features on train data
    f = open(cnn_train_merged_features, 'rb')
    cnn_train_merged_features = pickle.load(f)
    cnn_train_merged_features = cnn_train_merged_features.reshape((-1, 2048))
    f.close()

    # load cnn features on test data
    f = open(cnn_test_merged_features, 'rb')
    cnn_test_merged_features = pickle.load(f)
    cnn_test_merged_features = cnn_test_merged_features.reshape((-1, 2048))
    f.close()

    print('Load data')

    # load all the data to memory.
    # NOTE: you have to modify this to read the data
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images[:100, :, :]
    train_labels = (train_labels[:100] == 2).astype(np.uint8)
    test_images = test_images[:100, :, :]
    test_labels = (test_labels[:100] == 2).astype(np.uint8)

    print('Compute hand-craft features')
    # NOTE: I duplicate the process because this data only has one image input
    # so you have to modify the code to compute the features from each input image
    # then, concat them
    train_features = []
    for i in range(train_images.shape[0]):
        first_order_features = compute_handcraft_features(train_images[i])
        second_order_features = compute_haralick_features(train_images[i])
        
        features = np.concatenate([first_order_features, second_order_features, first_order_features, second_order_features])
        train_features.append(features)

    test_features = []
    for i in range(test_images.shape[0]):
        first_order_features = compute_handcraft_features(test_images[i])
        second_order_features = compute_haralick_features(test_images[i])

        features = np.concatenate([first_order_features, second_order_features, first_order_features, second_order_features])
        test_features.append(features)
    print(cnn_train_merged_features.shape, cnn_test_merged_features.shape)

    print('Compute hand-craft features')
    # merge the cnn features and hand-craft features
    train_features = np.concatenate([train_features, cnn_train_merged_features], axis=1)
    test_features = np.concatenate([test_features, cnn_test_merged_features], axis=1)

    # NOTE: you can adjust the max_iter, n_features_to_select, step to fit your data
    print('Optimization started.')
    estimator = SVR(kernel="linear", max_iter=200)  # init svm
    selector = RFE(estimator, n_features_to_select=100, step=2)  # init rfe
    selector = selector.fit(train_features, train_labels)  # optimize
    print('Optimization ended.')
    test_predicted = selector.predict(test_features)  # predict the output on test set

    precision_results = precision(test_labels, test_predicted)
    recall_results = recall(test_labels, test_predicted)

    print('Precision: {} - Recall: {}'.format(precision_results, recall_results))


def svm_stage_2_combine_2(cnn_train_merged_features='resnet50_train_merged_features.pkl', cnn_test_merged_features='resnet50_test_merged_features.pkl'):
    """This function is used to train and get the output results of the SVM (stage 2).
    For this strategy, the paper suggested to concat the features of the CNN and the features of stage-1 SVM instead of using the results of stage-1 SVM.
    Note, this strategy apply PCA on CNN features before the concatenation.

    Args:
        :param saved_cnn_name: the name to save the model.
    """
    print('Read CNN features')
    # load cnn features on train data
    f = open(cnn_train_merged_features, 'rb')
    cnn_train_merged_features = pickle.load(f)
    cnn_train_merged_features = cnn_train_merged_features.reshape((-1, 2048))
    f.close()

    # load cnn features on test data
    f = open(cnn_test_merged_features, 'rb')
    cnn_test_merged_features = pickle.load(f)
    cnn_test_merged_features = cnn_test_merged_features.reshape((-1, 2048))
    f.close()

    # NOTE: n_components need to be modify to 200
    # however, in this example, I only a very small data. So, I have to set
    # a small number of components
    print('Apply PCA on CNN features')
    # apply pca
    pca = PCA(n_components=50, svd_solver='full')
    cnn_train_merged_features = pca.fit_transform(cnn_train_merged_features)
    cnn_test_merged_features = pca.transform(cnn_test_merged_features)

    print('Load data')
    # load all the data to memory.
    # NOTE: you have to modify this to read the data
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images[:100, :, :]
    train_labels = (train_labels[:100] == 2).astype(np.uint8)
    test_images = test_images[:100, :, :]
    test_labels = (test_labels[:100] == 2).astype(np.uint8)

    print('Compute hand-craft features')
    # NOTE: I duplicate the process because this data only has one image input
    # so you have to modify the code to compute the features from each input image
    # then, concat them
    train_features = []
    for i in range(train_images.shape[0]):
        first_order_features = compute_handcraft_features(train_images[i])
        second_order_features = compute_haralick_features(train_images[i])
        
        features = np.concatenate([first_order_features, second_order_features, first_order_features, second_order_features])
        train_features.append(features)

    test_features = []
    for i in range(test_images.shape[0]):
        first_order_features = compute_handcraft_features(test_images[i])
        second_order_features = compute_haralick_features(test_images[i])

        features = np.concatenate([first_order_features, second_order_features, first_order_features, second_order_features])
        test_features.append(features)

    # merge the cnn features and hand-craft features
    train_features = np.concatenate([train_features, cnn_train_merged_features], axis=1)
    test_features = np.concatenate([test_features, cnn_test_merged_features], axis=1)

    # NOTE: you can adjust the max_iter, n_features_to_select, step to fit your data
    print('Optimization started.')
    estimator = SVR(kernel="linear", max_iter=200000)  # init svm
    selector = RFE(estimator, n_features_to_select=100, step=2)  # init rfe
    selector = selector.fit(train_features, train_labels)  # optimize
    print('Optimization ended.')
    test_predicted = selector.predict(test_features)  # predict the output on test set

    precision_results = precision(test_labels, test_predicted)
    recall_results = recall(test_labels, test_predicted)

    print('Precision: {} - Recall: {}'.format(precision_results, recall_results))


def svm_stage_2_combine_3(cnn_train_predictions='resnet50_train_prediction.pkl', cnn_test_predictions='resnet50_test_prediction.pkl', svm_train_predictions='svm_train_prediction.pkl', svm_test_predictions='svm_test_prediction.pkl'):
    """This function is used to train and get the output results of the SVM (stage 2).
    For this strategy, the paper suggested to concat the prediction of the CNN and the prediction of stage-1 SVM.

    Args:
        :param saved_cnn_name: the name to save the model.
    """
    print('Read predictions from the CNN and SVM stage-1')
    # load cnn predictions on train data
    f = open(cnn_train_predictions, 'rb')
    cnn_train_predictions = pickle.load(f)
    cnn_train_predictions = cnn_train_predictions.reshape((-1, 1))
    f.close()

    # load cnn predictions on test data
    f = open(cnn_test_predictions, 'rb')
    cnn_test_predictions = pickle.load(f)
    cnn_test_predictions = cnn_test_predictions.reshape((-1, 1))
    f.close()

    # load svm predictions on train data
    f = open(svm_train_predictions, 'rb')
    svm_train_predictions = pickle.load(f).reshape((-1, 1))
    f.close()

    # load svm predictions on test data
    f = open(svm_test_predictions, 'rb')
    svm_test_predictions = pickle.load(f).reshape((-1, 1))
    f.close()

    print('Read data')
    # load all the data to memory.
    # NOTE: you have to modify this to read the data
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images[:100, :, :]
    train_labels = (train_labels[:100] == 2).astype(np.uint8)
    test_images = test_images[:100, :, :]
    test_labels = (test_labels[:100] == 2).astype(np.uint8)

    # merge the cnn features and hand-craft features
    train_features = np.concatenate([svm_train_predictions, cnn_train_predictions], axis=1)
    test_features = np.concatenate([svm_test_predictions, cnn_test_predictions], axis=1)

    # NOTE: you can adjust the max_iter, n_features_to_select, step to fit your data
    print('Optimization started.')
    estimator = SVR(kernel="linear", max_iter=20000)  # init svm
    selector = RFE(estimator, n_features_to_select=8, step=5)  # init rfe
    selector = selector.fit(train_features, train_labels)  # optimize
    print('Optimization ended.')
    test_predicted = selector.predict(test_features)  # predict the output on test set

    precision_results = precision(test_labels, test_predicted)
    recall_results = recall(test_labels, test_predicted)

    print('Precision: {} - Recall: {}'.format(precision_results, recall_results))
