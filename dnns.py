import tensorflow as tf
from tensorflow import keras

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def reduction_block(x, out_filters):
    x = keras.layers.Conv2D(filters=out_filters, kernel_size=(1, 1),
                            strides=1, padding='same', use_bias=False)(x)

    x = keras.layers.BatchNormalization(momentum=0.1)(x)
    return keras.layers.ReLU()(x)


def get_app_net(net_name, input_shape):
    if net_name == 'densenet':
        base_model = keras.applications.densenet.DenseNet121(weights='imagenet',
                                                             include_top=False, input_shape=input_shape)
    elif net_name == 'mobilenet_v2':
        base_model = keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False,
                                                                 input_shape=input_shape)
    elif net_name == 'mobilenet':
        base_model = keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=False,
                                                            input_shape=input_shape)
    else:
        raise AttributeError(f'{net_name} is unknown')

    return base_model


def setup_network(input_shape, net_name, trainable=False):
    base_model = get_app_net(net_name, input_shape)
    base_model.trainable = trainable
    return base_model


def scene_net(image_size, base_network=None, classes=1, out_filters=0, dropout=0, activation='softmax', flatten=True):
    base_model = setup_network(image_size, base_network)

    inputs = keras.layers.Input(shape=image_size, dtype='float32')
    x = keras.layers.LayerNormalization()(inputs)

    x = base_model(x, training=False)
    if out_filters:
        x = reduction_block(x, out_filters)

    if flatten:
        x = keras.layers.Flatten()(x)
    else:
        x = keras.layers.GlobalAveragePooling2D()(x)
    if dropout:
        x = keras.layers.Dropout(dropout)(x)  # Regularize with dropout

    outputs = keras.layers.Dense(classes, activation=activation)(x)
    return keras.Model(inputs, outputs)
