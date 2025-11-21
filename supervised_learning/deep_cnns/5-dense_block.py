#!/usr/bin/env python3
"""
Builds a dense block as described in
'Densely Connected Convolutional Networks' (DenseNet).

Bottleneck design (DenseNet-B) is used:
For each layer in the block:
    - BatchNorm -> ReLU -> Conv1x1 (4 * growth_rate filters)
    - BatchNorm -> ReLU -> Conv3x3 (growth_rate filters)
Output of each layer is concatenated with input and passed to the next layer
"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block.

    Parameters
    ----------
    X : keras layer / tensor
        Output from previous layer.
    nb_filters : int
        Number of filters in X.
    growth_rate : int
        Growth rate (k) for the dense block.
    layers : int
        Number of layers to append in the dense block.

    Returns
    -------
    (X, nb_filters) :
        X : the concatenated output of the dense block
        nb_filters : the updated number of filters after the block
    """
    he_init = K.initializers.he_normal(seed=0)

    concat_feat = X
    for i in range(layers):
        # Bottleneck 1x1 conv (4 * growth_rate filters)
        bn1 = K.layers.BatchNormalization(axis=3)(concat_feat)
        act1 = K.layers.Activation('relu')(bn1)
        conv1 = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer=he_init
        )(act1)

        # 3x3 conv (growth_rate filters)
        bn2 = K.layers.BatchNormalization(axis=3)(conv1)
        act2 = K.layers.Activation('relu')(bn2)
        conv2 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=he_init
        )(act2)

        # Concatenate the new feature maps with the previous features
        concat_feat = K.layers.Concatenate(axis=3)([concat_feat, conv2])
        nb_filters += growth_rate

    return concat_feat, nb_filters
