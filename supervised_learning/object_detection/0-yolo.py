#!/usr/bin/env python3
"""
YOLO v3 Object Detection Module

This module defines the Yolo class used to perform object detection
using the YOLO v3 algorithm and a pre-trained Darknet Keras model.
"""

import tensorflow as tf


class Yolo:
    """
    Yolo class for object detection using YOLO v3
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor

        Parameters
        ----------
        model_path : str
            Path to the Darknet Keras model (.h5 file)

        classes_path : str
            Path to the file containing class names, one per line,
            ordered by index

        class_t : float
            Box score threshold for the initial filtering step

        nms_t : float
            Intersection over Union (IoU) threshold for non-max suppression

        anchors : numpy.ndarray
            Array of shape (outputs, anchor_boxes, 2) containing
            all anchor boxes:
                - outputs: number of model outputs
                - anchor_boxes: number of anchors per output
                - 2: [anchor_width, anchor_height]
        """

        # Load the YOLO model using TensorFlow Keras
        # compile=False is required for compatibility with older models
        self.model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        # Load class names from file
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f if line.strip()]

        # Store thresholds and anchor boxes
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
