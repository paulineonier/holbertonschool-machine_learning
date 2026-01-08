#!/usr/bin/env python3
"""
Class Yolo with methods:
- process_outputs
- filter_boxes
- non_max_suppression
- load_images
- preprocess_images
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
        """
        self.model = tf.keras.models.load_model(model_path, compile=False)

        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process Darknet model outputs"""
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

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filters YOLO boxes based on object and class confidence"""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, conf, class_prob in zip(boxes, box_confidences, box_class_probs):
            scores = conf * class_prob
            class_ids = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)
            mask = class_scores >= self.class_t

            filtered_boxes.append(box[mask])
            box_classes.append(class_ids[mask])
            box_scores.append(class_scores[mask])

        if filtered_boxes:
            filtered_boxes = np.concatenate(filtered_boxes, axis=0)
            box_classes = np.concatenate(box_classes, axis=0)
            box_scores = np.concatenate(box_scores, axis=0)
        else:
            filtered_boxes = np.array([])
            box_classes = np.array([])
            box_scores = np.array([])

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Applies Non-Maximum Suppression (NMS)"""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            idxs = np.where(box_classes == cls)
            cls_boxes = filtered_boxes[idxs]
            cls_scores = box_scores[idxs]

            x1 = cls_boxes[:, 0]
            y1 = cls_boxes[:, 1]
            x2 = cls_boxes[:, 2]
            y2 = cls_boxes[:, 3]

            areas = (x2 - x1) * (y2 - y1)
            order = np.argsort(-cls_scores)

            keep = []

            while order.size > 0:
                i = order[0]
                keep.append(i)

                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h
                iou = inter / (areas[i] + areas[order[1:]] - inter)

                order = order[1:][iou < self.nms_t]

            box_predictions.append(cls_boxes[keep])
            predicted_box_classes.append(np.full(len(keep), cls))
            predicted_box_scores.append(cls_scores[keep])

        box_predictions = np.concatenate(box_predictions, axis=0)
        predicted_box_classes = np.concatenate(predicted_box_classes, axis=0)
        predicted_box_scores = np.concatenate(predicted_box_scores, axis=0)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """Loads images from a folder"""
        images = []
        image_paths = []

        extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        for ext in extensions:
            for path in glob.glob(os.path.join(folder_path, ext)):
                image = cv2.imread(path)
                if image is not None:
                    images.append(image)
                    image_paths.append(path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocess images for YOLO model
        - Resize with INTER_CUBIC
        - Rescale pixels to [0,1]
        Returns: (pimages, image_shapes)
        """
        pimages = []
        image_shapes = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        for img in images:
            image_shapes.append([img.shape[0], img.shape[1]])
            resized = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_CUBIC)
            normalized = resized / 255.0
            pimages.append(normalized)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes
