#!/usr/bin/env python3
"""
0-transfer.py

Train a CIFAR-10 classifier using transfer learning from a Keras Application.
This script uses an application model from tensorflow.keras.applications,
scales CIFAR-10 images from 32x32 up to the required input size using a
Lambda layer, computes frozen bottleneck features once, then trains a small
classifier on top and saves the trained, compiled model as cifar10.h5.

The module provides:
- preprocess_data(X, Y)
- When run as a script, trains and saves cifar10.h5
"""

from tensorflow import keras as K

__doc__ = __doc__


def preprocess_data(X, Y):
    """
    Preprocess CIFAR-10 data for the transfer learning model.

    Arguments:
    X: numpy.ndarray of shape (m, 32, 32, 3)
    Y: numpy.ndarray of shape (m,)

    Returns:
    X_p: preprocessed images resized to 224x224
    Y_p: labels reshaped
    """
    import numpy as np

    X = X.astype('float32')
    inputs = K.Input(shape=(32, 32, 3))
    resized = K.layers.Lambda(
        lambda img: K.backend.resize_images(img, 7, 7, "channels_last")
    )(inputs)
    resize_model = K.Model(inputs=inputs, outputs=resized)

    m = X.shape[0]
    batch = 256
    X_resized = np.zeros((m, 224, 224, 3), dtype='float32')
    for i in range(0, m, batch):
        j = min(i + batch, m)
        X_resized[i:j] = resize_model.predict(X[i:j], verbose=0)

    try:
        preprocess_fn = K.applications.efficientnet.preprocess_input
    except Exception:
        preprocess_fn = K.applications.mobilenet_v2.preprocess_input

    X_p = preprocess_fn(X_resized)
    Y_p = Y.reshape(-1,)
    return X_p, Y_p


class Trainer:
    """
    Trainer class encapsulating training and saving logic.
    """

    def __init__(self):
        """Initialize default hyperparameters."""
        self.batch_size = 128
        self.epochs_top = 20
        self.epochs_finetune = 10
        self.input_size = (224, 224, 3)
        self.num_classes = 10
        self.model_filename = 'cifar10.h5'

    def build_base(self):
        """
        Build a frozen EfficientNet base model.
        """
        base = K.applications.efficientnet.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_size,
            pooling='avg'
        )
        base.trainable = False
        inputs = K.Input(shape=self.input_size)
        outputs = base(inputs)
        return K.Model(inputs=inputs, outputs=outputs)

    def build_top(self, feature_shape):
        """
        Build the classifier head.
        """
        inputs = K.Input(shape=feature_shape)
        x = K.layers.Dense(512, activation='relu')(inputs)
        x = K.layers.Dropout(0.4)(x)
        x = K.layers.Dense(256, activation='relu')(x)
        x = K.layers.Dropout(0.3)(x)
        outputs = K.layers.Dense(self.num_classes, activation='softmax')(x)
        return K.Model(inputs=inputs, outputs=outputs)

    def train(self):
        """
        Run full training pipeline:
        preprocess → bottleneck → train top → fine-tune → save model.
        """
        (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
        Y_train = Y_train.reshape(-1,)
        Y_test = Y_test.reshape(-1,)

        X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
        X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

        base_model = self.build_base()
        train_features = base_model.predict(
            X_train_p, batch_size=self.batch_size, verbose=1
        )
        val_features = base_model.predict(
            X_test_p, batch_size=self.batch_size, verbose=1
        )

        top_model = self.build_top(train_features.shape[1:])
        top_model.compile(
            optimizer=K.optimizers.Adam(1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        early = K.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
        reduce_lr = K.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )

        top_model.fit(
            train_features,
            Y_train_p,
            validation_data=(val_features, Y_test_p),
            epochs=self.epochs_top,
            batch_size=self.batch_size,
            callbacks=[early, reduce_lr],
            verbose=1
        )

        base_full = K.applications.efficientnet.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_size,
            pooling='avg'
        )

        for layer in base_full.layers[:-20]:
            layer.trainable = False
        for layer in base_full.layers[-20:]:
            layer.trainable = True

        inputs = K.Input(shape=self.input_size)
        x = base_full(inputs)
        outputs = top_model(x)
        combined = K.Model(inputs=inputs, outputs=outputs)

        combined.compile(
            optimizer=K.optimizers.Adam(1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        combined.fit(
            X_train_p,
            Y_train_p,
            validation_data=(X_test_p, Y_test_p),
            epochs=self.epochs_finetune,
            batch_size=self.batch_size,
            callbacks=[early, reduce_lr],
            verbose=1
        )

        combined.save(self.model_filename, include_optimizer=True)
        combined.evaluate(X_test_p, Y_test_p, batch_size=self.batch_size, verbose=1)


if __name__ == '__main__':
    Trainer().train()
