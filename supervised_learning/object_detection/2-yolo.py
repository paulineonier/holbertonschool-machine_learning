#!/usr/bin/env python3
"""
Class Yolo with methods process_outputs and filter_boxes
"""

import numpy as np
import tensorflow as tf


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
        # Load Keras model
        self.model = tf.keras.models.load_model(model_path, compile=False)
        
        # Load class names
        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]
        
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Process Darknet model outputs

        outputs: list of numpy.ndarrays from Darknet model for a single image
        image_size: numpy.ndarray containing original image size [height, width]

        Returns: (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            # Split the output
            t_xy = output[..., 0:2]
            t_wh = output[..., 2:4]
            box_conf = output[..., 4:5]
            class_probs = output[..., 5:]

            # Sigmoid for t_xy, box_confidence, class probabilities
            sigmoid_xy = 1 / (1 + np.exp(-t_xy))
            sigmoid_conf = 1 / (1 + np.exp(-box_conf))
            sigmoid_class = 1 / (1 + np.exp(-class_probs))

            # Create grid for cx, cy
            grid_x = np.arange(grid_w)
            grid_y = np.arange(grid_h)
            cx, cy = np.meshgrid(grid_x, grid_y)
            cx = np.expand_dims(cx, axis=-1)
            cy = np.expand_dims(cy, axis=-1)

            # Adjust xy to original scale
            bx = (sigmoid_xy[..., 0] + cx) / grid_w
            by = (sigmoid_xy[..., 1] + cy) / grid_h

            # Adjust width/height using anchors
            pw = self.anchors[i, :, 0].reshape((1, 1, anchor_boxes))
            ph = self.anchors[i, :, 1].reshape((1, 1, anchor_boxes))
            bw = pw * np.exp(t_wh[..., 0]) / self.model.input.shape[1]
            bh = ph * np.exp(t_wh[..., 1]) / self.model.input.shape[2]

            # Convert to corner coordinates relative to original image
            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            box = np.stack([x1, y1, x2, y2], axis=-1)

            boxes.append(box)
            box_confidences.append(sigmoid_conf)
            box_class_probs.append(sigmoid_class)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters YOLO boxes based on object and class confidence

        boxes: list of numpy.ndarrays of shape (grid_h, grid_w, anchor_boxes, 4)
        box_confidences: list of numpy.ndarrays of shape (grid_h, grid_w, anchor_boxes, 1)
        box_class_probs: list of numpy.ndarrays of shape (grid_h, grid_w, anchor_boxes, classes)

        Returns: (filtered_boxes, box_classes, box_scores)
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, conf, class_prob in zip(boxes, box_confidences, box_class_probs):
            # Compute box scores: confidence * class probabilities
            scores = conf * class_prob  # shape: (grid_h, grid_w, anchors, classes)

            # Find the class with the maximum score for each box
            class_ids = np.argmax(scores, axis=-1)  # shape: (grid_h, grid_w, anchors)
            class_scores = np.max(scores, axis=-1)  # shape: (grid_h, grid_w, anchors)

            # Apply threshold
            mask = class_scores >= self.class_t

            # Filter boxes
            filtered_boxes.append(box[mask])
            box_classes.append(class_ids[mask])
            box_scores.append(class_scores[mask])

        # Concatenate all outputs
        if filtered_boxes:
            filtered_boxes = np.concatenate(filtered_boxes, axis=0)
            box_classes = np.concatenate(box_classes, axis=0)
            box_scores = np.concatenate(box_scores, axis=0)
        else:
            filtered_boxes = np.array([])
            box_classes = np.array([])
            box_scores = np.array([])

        return filtered_boxes, box_classes, box_scores
