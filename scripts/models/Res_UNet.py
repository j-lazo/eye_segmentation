import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


# model U-Net
def res_conv_block(x, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    skip = tf.keras.layers.Conv2D(num_filters, (3, 3), padding="same")(x)
    skip = tf.keras.layers.Activation("relu")(skip)
    skip = tf.keras.layers.BatchNormalization()(skip)

    x = tf.keras.layers.Conv2D(num_filters, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.Activation("relu")(x)

    return x


def build_model(input_size, num_filters = [64, 128, 256, 512]):
    #num_filters = [16, 32, 48, 64]
    
    inputs = tf.keras.Input((input_size, input_size, 3))  
    skip_x = []
    x = inputs

    ## Encoder
    for f in num_filters:
        x = res_conv_block(x, f)
        skip_x.append(x)
        x = tf.keras.layers.MaxPool2D((2, 2))(x)

    ## Bridge
    x = res_conv_block(x, num_filters[-1])

    num_filters.reverse()
    skip_x.reverse()
    ## Decoder
    for i, f in enumerate(num_filters):
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        xs = skip_x[i]
        x = tf.keras.layers.Concatenate()([x, xs])
        x = res_conv_block(x, f)

    ## Output
    x = tf.keras.layers.Conv2D(1, (1, 1), padding="same")(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    return Model(inputs, x)
