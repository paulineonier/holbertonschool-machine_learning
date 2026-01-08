#!/usr/bin/env python3
"""
Class Yolo with methods:
- process_outputs
- filter_boxes
- non_max_suppression
- static method load_images
"""

import numpy as np
import tensorflow as tf
import cv2
import glob
import os


class Yolo:
    """YOLO v3 object detection class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor
        model_path: path to Darknet Keras model
        classes_path: path to class names
        class_t: box score threshold
        nms_t: non-max suppression threshold
        anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
        """
        self.model = tf.keras.models.load_model(model_path, compile=False)

        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Process Darknet model outputs
        Returns: (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            t_xy = output[..., 0:2]
            t_wh = output[..., 2:4]
            box_conf = output[..., 4:5]
            class_probs = output[..., 5:]

            sigmoid_xy = 1 / (1 + np.exp(-t_xy))
            sigmoid_conf = 1 / (1 + np.exp(-box_conf))
            sigmoid_class = 1 / (1 + np.exp(-class_probs))

            grid_x = np.arange(grid_w)
            grid_y = np.arange(grid_h)
            cx, cy = np.meshgrid(grid_x, grid_y)
            cx = np.expand_dims(cx, axis=-1)
            cy = np.expand_dims(cy, axis=-1)

            bx = (sigmoid_xy[..., 0] + cx) / grid_w
            by = (sigmoid_xy[..., 1] + cy) / grid_h

            pw = self.anchors[i, :, 0].reshape((1, 1, anchor_boxes))
            ph = self.anchors[i, :, 1].reshape((1, 1, anchor_boxes))
            bw = pw * np.exp(t_wh[..., 0]) / self.model.input.shape[1]
            bh = ph * np.exp(t_wh[..., 1]) / self.model.input.shape[2]

            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            box = np.stack([x1, y1, x2, y2], axis=-1)

            boxes.append(box)
            box_confidences.append(sigmoid_conf)
            box_class_probs.append(sigmoid_class)
