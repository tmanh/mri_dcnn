import tensorflow as tf
import tensorflow.keras as keras

from googlenet import Inceptionv1


def create_stem(name, modality='vgg16', weights='imagenet'):
    """This function is used to create the based CNN for the deep network.

    Args:
        :param modality: the modality to use (resnet50, vgg16, googlenet).
    """

    # create the based network based on the modality
    if modality == 'vgg16':
        stem = tf.keras.applications.VGG16(include_top=False, weights=weights, classes=2, classifier_activation='sigmoid')
    elif modality == 'resnet50':
        stem = tf.keras.applications.ResNet50(include_top=False, weights=weights, classes=2)
    else:
        inception_builder = Inceptionv1()
        stem = inception_builder.build_inception()

    # modified the name to avoid name confliction
    stem._name = stem.name + '_' + name

    return stem


def create_network(image_size, image_dim, weights='imagenet', classifier_activation='softmax', modality='resnet50', fusion_mode='paper', combine_mode='c1', training=True):
    """This function is used to create the deep network.

    Args:
        :param image_size: the size of the input image.
        :param image_dim: the number of channels of the input image.
        :param n_classes: the number of classes need to be classified.
        :param classifier_activation: the activation function for the classification.
        :param modality: the modality to use (resnet50, vgg16, googlenet).
        :param fusion_mode: the fusion mode to use (concatenate, sum, paper, adc, t2wi) (using concatenation, element-wise sum, and the proposed one in the paper, only use adc, only use t2wi).
        :param combine_mode: c1, c2, c3. c1 just concat, c2 pca then concat, c3 concat prediction.
        :param training: indicate if we train the deep model or just use it as a feature extractors.
    """

    n_classes=2

    # define two sets of inputs
    input_1 = keras.Input(shape=(image_size, image_size, image_dim))
    input_2 = keras.Input(shape=(image_size, image_size, image_dim))

    # define base CNN nets
    branch_1 = create_stem('branch_1', modality=modality, weights=weights)
    branch_2 = create_stem('branch_2', modality=modality, weights=weights)

    # in case of concatenation strategy
    if fusion_mode == 'concatenate':
        n_feats_dense_1 = 2048      # define number of node in the FC layer

        cnn_x1 = branch_1(input_1)  # extract features by the base nets (ADC)
        cnn_x2 = branch_2(input_2)  # extract features by the base nets (T2WI)

        # global pooling
        cnn_x1 = tf.keras.layers.GlobalAveragePooling2D()(cnn_x1)
        cnn_x2 = tf.keras.layers.GlobalAveragePooling2D()(cnn_x2)

        # prediction
        x1 = keras.layers.Flatten(name='flatten_1')(cnn_x1)                                                       # reshape the features maps from N x H x W x C to N x (H * W * C)
        x1 = keras.layers.Dense(n_feats_dense_1, activation='relu', name='fc1_x1')(x1)                            # apply fc layer
        predict_1 = keras.layers.Dense(n_classes, activation=classifier_activation, name='classification_1')(x1)  # classification layers (ADC branch)

        x2 = keras.layers.Flatten(name='flatten_2')(cnn_x2)                                                       # reshape the features maps from N x H x W x C to N x (H * W * C)
        x2 = keras.layers.Dense(n_feats_dense_1, activation='relu', name='fc1_x2')(x2)                            # apply fc layer
        predict_2 = keras.layers.Dense(n_classes, activation=classifier_activation, name='classification_2')(x2)  # classification layers (T2WI branch)

        # fusion
        x = keras.layers.Concatenate(axis=1)([x1, x2])                                                            # concat the features
        predict = keras.layers.Dense(n_classes, activation=classifier_activation, name='classification')(x)       # classification layers (fusion)

        # return the data to match with mode (training, testing)
        # in testing mode, the output list also has the concatenated features
        if training:
            model = keras.Model(inputs=(input_1, input_2), outputs=(predict_1, predict_2, predict))
        else:
            model = keras.Model(inputs=(input_1, input_2), outputs=(predict_1, predict_2, predict, x))
    elif fusion_mode == 'sum':
        n_feats_dense_1 = 2048      # define number of node in the FC layer

        cnn_x1 = branch_1(input_1)  # extract features by the base nets (ADC)
        cnn_x2 = branch_2(input_2)  # extract features by the base nets (T2WI)

        # global pooling
        cnn_x1 = tf.keras.layers.GlobalAveragePooling2D()(cnn_x1)
        cnn_x2 = tf.keras.layers.GlobalAveragePooling2D()(cnn_x2)

        # prediction
        x1 = keras.layers.Flatten(name='flatten_1')(cnn_x1)                                                       # reshape the features maps from N x H x W x C to N x (H * W * C)
        x1 = keras.layers.Dense(n_feats_dense_1, activation='relu', name='fc1_x1')(x1)                            # apply fc layer
        predict_1 = keras.layers.Dense(n_classes, activation=classifier_activation, name='classification_1')(x1)  # classification layers (ADC branch)

        x2 = keras.layers.Flatten(name='flatten_2')(cnn_x2)                                                       # reshape the features maps from N x H x W x C to N x (H * W * C)
        x2 = keras.layers.Dense(n_feats_dense_1, activation='relu', name='fc1_x2')(x2)                            # apply fc layer
        predict_2 = keras.layers.Dense(n_classes, activation=classifier_activation, name='classification_2')(x2)  # classification layers (T2WI branch)

        # fusion
        fused = cnn_x1 + cnn_x2
        x = keras.layers.Flatten(name='flatten')(fused)                                                           # merge feature by element-wise sum
        x = keras.layers.Dense(n_feats_dense_1, activation='relu', name='fc1')(x)                                 # apply fc layer
        predict = keras.layers.Dense(n_classes, activation=classifier_activation, name='classification')(x)       # classification layers (fusion)

        # return the data to match with mode (training, testing)
        # in testing mode, the output list also has the element-wise sum features
        if training:
            model = keras.Model(inputs=(input_1, input_2), outputs=(predict_1, predict_2, predict))
        else:
            model = keras.Model(inputs=(input_1, input_2), outputs=(predict_1, predict_2, predict, x))
    elif fusion_mode == 'adc':
        n_feats_dense_1 = 2048      # define number of node in the FC layer

        cnn_x1 = branch_1(input_1)  # extract features by the base nets (ADC)
        cnn_x1 = tf.keras.layers.GlobalAveragePooling2D()(cnn_x1)

        x1 = keras.layers.Flatten(name='flatten')(cnn_x1)                                                       # reshape the features maps from N x H x W x C to N x (H * W * C)
        x1 = keras.layers.Dense(n_feats_dense_1, activation='relu', name='fc1_x1')(x1)                           # apply fc layer
        predict_1 = keras.layers.Dense(n_classes, activation=classifier_activation, name='classification')(x1)   # classification layers (T2WI branch)

        # return the data to match with mode (training, testing)
        # in testing mode, the output list also has the features of the last fc layer
        if training:
            model = keras.Model(inputs=(input_1, input_2), outputs=(predict_1, predict_1, predict_1))
        else:
            model = keras.Model(inputs=(input_1, input_2), outputs=(predict_1, x1))
    elif fusion_mode == 't2wi':
        n_feats_dense_1 = 2048      # define number of node in the FC layer

        cnn_x2 = branch_2(input_2)  # extract features by the base nets (T2WI)
        cnn_x2 = tf.keras.layers.GlobalAveragePooling2D()(cnn_x2)

        x2 = keras.layers.Flatten(name='flatten')(cnn_x2)                                                       # reshape the features maps from N x H x W x C to N x (H * W * C)
        x2 = keras.layers.Dense(n_feats_dense_1, activation='relu', name='fc1_x1')(x2)                          # apply fc layer
        predict_2 = keras.layers.Dense(n_classes, activation=classifier_activation, name='classification')(x2)  # classification layers (T2WI branch)

        # return the data to match with mode (training, testing)
        # in testing mode, the output list also has the features of the last fc layer
        if training:
            model = keras.Model(inputs=(input_1, input_2), outputs=(predict_2, predict_2, predict_2))
        else:
            model = keras.Model(inputs=(input_1, input_2), outputs=[predict_2, x2])
    else:  # only for resnet50 and binary classification (non-cancer or cancer)
        x1 = branch_1(input_1)  # extract features by the base nets (ADC)
        x2 = branch_2(input_2)  # extract features by the base nets (T2WI)

        conv1 = keras.layers.Conv2D(1, 1)(x1)
        conv2 = keras.layers.Conv2D(1, 1)(x2)

        pool_x1 = tf.keras.layers.GlobalAveragePooling2D()(conv1)  # apply global pooling
        pool_x2 = tf.keras.layers.GlobalAveragePooling2D()(conv2)  # apply global pooling

        predict_1 = tf.keras.activations.sigmoid(pool_x1)       # apply sigmoid
        predict_2 = tf.keras.activations.sigmoid(pool_x2)       # apply sigmoid

        # compute the similarity
        similarity = tf.math.squared_difference(tf.keras.activations.sigmoid(x1), tf.keras.activations.sigmoid(x2))
        similarity = tf.reduce_mean(tf.reduce_mean(similarity * 0.5, axis=1), axis=1)

        # compute the weight to combine for the final prediction
        w = tf.random.uniform([1], minval=0, maxval=1)

        predict = predict_1 * w + predict_2 * (1 - w)  # merge feature maps

        predict_1 = keras.layers.Dense(n_classes, activation=classifier_activation, name='classification_1')(predict_1)  # classification layers (ADC branch)
        predict_2 = keras.layers.Dense(n_classes, activation=classifier_activation, name='classification_2')(predict_2)  # classification layers (T2WI branch)
        predict = keras.layers.Dense(n_classes, activation=classifier_activation, name='classification')(predict)

        # note: in the paper, they did not state the way the produce this merge features.
        # they only described that the feature vector is 2048. Which could only be produce by the GlobalAveragePooling2D on the feature maps of the last conv layer
        merge_features = tf.keras.layers.GlobalAveragePooling2D()(x1) + tf.keras.layers.GlobalAveragePooling2D()(x2)

        # return the data to match with mode (training, testing)
        # in testing mode, the output list also has the features of the fused features
        if training:
            model = keras.Model(inputs=(input_1, input_2), outputs=(predict_1, predict_2, predict, similarity))
        else:
            model = keras.Model(inputs=(input_1, input_2), outputs=[predict_1, predict_2, predict, merge_features])
    return model
