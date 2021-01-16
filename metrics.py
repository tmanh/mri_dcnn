import numpy as np
import tensorflow.keras.backend as keras_backend


def recall_m(y_true, y_pred):
    """ function to compute recall

    Args:
        :param y_true: ground-truth.
        :param y_pred: prediction
    """
    # compute the true positives
    true_positives = keras_backend.sum(keras_backend.round(keras_backend.clip(y_true * y_pred, 0, 1)))

    # compute all the positives
    possible_positives = keras_backend.sum(keras_backend.round(keras_backend.clip(y_true, 0, 1)))

    # compute recall
    recall = true_positives / (possible_positives + keras_backend.epsilon())
    return recall


def precision_m(y_true, y_pred):
    """ function to compute precision

    Args:
        :param y_true: ground-truth.
        :param y_pred: prediction
    """
    # compute the true positives
    true_positives = keras_backend.sum(keras_backend.round(keras_backend.clip(y_true * y_pred, 0, 1)))

    # compute all correct predictions
    predicted_positives = keras_backend.sum(keras_backend.round(keras_backend.clip(y_pred, 0, 1)))

    # compute precision
    precision = true_positives / (predicted_positives + keras_backend.epsilon())
    return precision


def similarity_loss(_, similarity):
    """ function to compute the similarity

    Args:
        :param similarity: the similarity that have been computed in the network
    """
    return similarity


def recall(y_true, y_pred):
    """ function to compute recall

    Args:
        :param y_true: ground-truth.
        :param y_pred: prediction
    """
    # compute the true positives
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))

    # compute all the positives
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))

    # compute recall
    recall = true_positives / (possible_positives + 1e-5)
    return recall


def precision(y_true, y_pred):
    """ function to compute precision

    Args:
        :param y_true: ground-truth.
        :param y_pred: prediction
    """
    # compute the true positives
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))

    # compute all correct predictions
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))

    # compute precision
    precision = true_positives / (predicted_positives +  + 1e-5)
    return precision
