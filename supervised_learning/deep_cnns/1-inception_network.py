#!/usr/bin/env python3
"""
Builds the Inception network (GoogLeNet, 2014)
"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Builds the Inception network as described in
    'Going Deeper with Convolutions' (2014)
    Returns: the keras model
    """
    input_layer = K.Input(shape=(224, 224, 3))

    # Initial layers
    conv1 = K.layers.Conv2D(64, (7, 7), strides=(2, 2),
                            padding='same', activation='relu')(input_layer)
    pool1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                                  padding='same')(conv1)

    conv2 = K.layers.Conv2D(64, (1, 1), padding='same',
                            activation='relu')(pool1)
    conv3 = K.layers.Conv2D(192, (3, 3), padding='same',
                            activation='relu')(conv2)
    pool2 = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                                  padding='same')(conv3)

    # Inception blocks
    inception_3a = inception_block(pool2, [64, 96, 128, 16, 32, 32])
    inception_3b = inception_block(inception_3a, [128, 128, 192, 32, 96, 64])
    pool3 = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                                  padding='same')(inception_3b)

    inception_4a = inception_block(pool3, [192, 96, 208, 16, 48, 64])
    inception_4b = inception_block(inception_4a, [160, 112, 224, 24, 64, 64])
    inception_4c = inception_block(inception_4b, [128, 128, 256, 24, 64, 64])
    inception_4d = inception_block(inception_4c, [112, 144, 288, 32, 64, 64])
    inception_4e = inception_block(inception_4d, [256, 160, 320, 32, 128, 128])
    pool4 = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                                  padding='same')(inception_4e)

    inception_5a = inception_block(pool4, [256, 160, 320, 32, 128, 128])
    inception_5b = inception_block(inception_5a, [384, 192, 384, 48, 128, 128])

    # Final layers
    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         strides=(1, 1),
                                         padding='valid')(inception_5b)
    dropout = K.layers.Dropout(0.4)(avg_pool)
    output_layer = K.layers.Dense(1000, activation='softmax')(dropout)

    model = K.models.Model(inputs=input_layer, outputs=output_layer)
    return model
