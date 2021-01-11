import tensorflow as tf


def create_stem(modality='vgg16'):
    """This function is used to create the based CNN for the deep network.

    Args:
        :param modality: the modality to use (resnet50, vgg16, googlenet).
    """

    if modality == 'vgg16':
        stem = tf.keras.applications.VGG16(include_top=False, weights='imagenet', classes=2, classifier_activation='softmax')
    elif modality == 'resnet50':
        stem = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', classes=2)
    else:
        pass

    return stem
