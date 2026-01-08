#!/usr/bin/env python3
"""
Yolo class for object detection with Darknet model
"""

import os
import cv2
import numpy as np
import tensorflow as tf

class Yolo:
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialise le modèle YOLO
        
        Parameters
        ----------
        model_path : str
            Chemin vers le modèle Keras (.h5)
        classes_path : str
            Chemin vers le fichier de classes
        class_t : float
            Seuil de confiance pour filtrer les boxes
        nms_t : float
            Seuil pour non-max suppression
        anchors : numpy.ndarray
            Boîtes d’ancrage (3, 3, 2)
        """
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Convertit les sorties du modèle en bounding boxes, confidences et class_probs
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_h, image_w = image_size
        for output in outputs:
            grid_h, grid_w, anchors_num, _ = output.shape

            t_xy = output[..., :2]
            t_wh = output[..., 2:4]
            conf = self.sigmoid(output[..., 4:5])
            class_probs = self.sigmoid(output[..., 5:])

            # Créer une grille
            grid_x = np.arange(grid_w)
            grid_y = np.arange(grid_h)
            cx, cy = np.meshgrid(grid_x, grid_y)
            cx = cx[..., np.newaxis]
            cy = cy[..., np.newaxis]

            # Calculer les offsets
            bx = (self.sigmoid(t_xy[..., 0]) + cx) / grid_w
            by = (self.sigmoid(t_xy[..., 1]) + cy) / grid_h

            pw = self.anchors[..., 0] / self.model.input.shape[1]
            ph = self.anchors[..., 1] / self.model.input.shape[2]

            bw = pw * np.exp(t_wh[..., 0])
            bh = ph * np.exp(t_wh[..., 1])

            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
            box_confidences.append(conf)
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filtrer les boxes selon le seuil de confiance
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for b, conf, probs in zip(boxes, box_confidences, box_class_probs):
            scores = conf * probs
            class_ids = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            mask = class_scores >= self.class_t
            filtered_boxes.append(b[mask])
            box_classes.append(class_ids[mask])
            box_scores.append(class_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applique la suppression non maximale pour supprimer les boxes qui se chevauchent
        """
        boxes_out = []
        classes_out = []
        scores_out = []

        unique_classes = np.unique(box_classes)
        for cls in unique_classes:
            idxs = np.where(box_classes == cls)[0]
            cls_boxes = filtered_boxes[idxs]
            cls_scores = box_scores[idxs]

            x1 = cls_boxes[:, 0]
            y1 = cls_boxes[:, 1]
            x2 = cls_boxes[:, 2]
            y2 = cls_boxes[:, 3]

            areas = (x2 - x1) * (y2 - y1)
            order = cls_scores.argsort()[::-1]

            while order.size > 0:
                i = order[0]
                boxes_out.append(cls_boxes[i])
                classes_out.append(cls)
                scores_out.append(cls_scores[i])

                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0, xx2 - xx1)
                h = np.maximum(0, yy2 - yy1)
                inter = w * h
                ovr = inter / (areas[i] + areas[order[1:]] - inter)

                inds = np.where(ovr <= self.nms_t)[0]
                order = order[inds + 1]

        return np.array(boxes_out), np.array(classes_out), np.array(scores_out)

    @staticmethod
    def load_images(folder_path):
        """
        Charge toutes les images dans un dossier
        """
        images = []
        image_paths = []

        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
                image_paths.append(path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Redimensionne et normalise les images
        """
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]
        pimages = []
        image_shapes = []

        for img in images:
            image_shapes.append([img.shape[0], img.shape[1]])
            resized = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_CUBIC)
            pimages.append(resized / 255.)

        return np.array(pimages), np.array(image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Affiche l'image avec les boxes, classes et scores
        """
        for box, cls, score in zip(boxes, box_classes, box_scores):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text = f"{self.class_names[cls]} {score:.2f}"
            cv2.putText(image, text, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)
        if key == ord('s'):
            if not os.path.exists("detections"):
                os.makedirs("detections")
            cv2.imwrite(os.path.join("detections", file_name), image)
        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Fait des prédictions sur toutes les images dans le dossier
        """
        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)
        predictions = []

        for i, img in enumerate(pimages):
            output = self.model.predict(np.expand_dims(img, axis=0))
            if not isinstance(output, list):
                output = [output]

            boxes, box_confidences, box_class_probs = self.process_outputs(output, image_shapes[i])
            f_boxes, b_classes, b_scores = self.filter_boxes(boxes, box_confidences, box_class_probs)
            nms_boxes, nms_classes, nms_scores = self.non_max_suppression(f_boxes, b_classes, b_scores)

            file_name = os.path.basename(image_paths[i])
            self.show_boxes(images[i], nms_boxes, nms_classes, nms_scores, file_name)

            predictions.append((nms_boxes, nms_classes, nms_scores))

        return predictions, image_paths
