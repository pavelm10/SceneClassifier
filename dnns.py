import tensorflow as tf

from keras.layers import Conv2D, BatchNormalization, PReLU, Input, Dropout, Flatten, Dense
from keras.models import Model

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def conv_block(dinput, kernels, kernel_size, stride):
    skip_conn = None
    convolved = Conv2D(filters=kernels, kernel_size=kernel_size,
                       strides=stride, padding='same', use_bias=False)(dinput)

    convolved = BatchNormalization(momentum=0.1)(convolved)
    convolved = PReLU(shared_axes=[1, 2])(convolved)

    return convolved, skip_conn


def scenenet_v1(ch_num=3, init_kernels=32, kernel_size=3, classes=1, activation='sigmoid', conv_reps=2):
    inp = Input(shape=(None, None, ch_num), dtype='float32')
    cur_kernels = init_kernels
    ks = kernel_size
    x = conv_block(inp, kernels=cur_kernels, kernel_size=(ks, ks), stride=1)  # 1/1
    cur_kernels *= 2
    for _ in range(conv_reps):
        x = conv_block(x, kernels=cur_kernels, kernel_size=(ks, ks), stride=2)  # 1/n
        cur_kernels *= 2

    x = Flatten()(x)

    out = Dense(classes, activation=activation)(x)
    model = Model(inputs=inp, outputs=out)
    return model
