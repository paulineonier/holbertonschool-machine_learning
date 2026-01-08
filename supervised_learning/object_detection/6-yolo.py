#!/usr/bin/env python3
import numpy as np
import cv2
import os
import tensorflow as tf


class Yolo:
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialise le modèle YOLO"""
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def load_images(folder_path):
        """Charge toutes les images d’un dossier"""
        images = []
        image_paths = []
        for file_name in os.listdir(folder_path):
            path = os.path.join(folder_path, file_name)
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
                image_paths.append(path)
        return images, image_paths

    def preprocess_images(self, images):
        """Redimensionne et normalise les images"""
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]
        pimages = []
        image_shapes = []

        for img in images:
            image_shapes.append([img.shape[0], img.shape[1]])
            resized = cv2.resize(img,
                                 (input_w, input_h), interpolation=cv2.INTER_CUBIC)
            pimages.append(resized / 255.0)

        return np.array(pimages), np.array(image_shapes)

    def process_outputs(self, outputs, image_size):
        """Transforme sorties du modèle en boîtes, scores et probabilités"""
        boxes_list = []
        box_confidences = []
        box_class_probs = []

        for output in outputs:
            grid_h, grid_w, anchors_num, _ = output.shape
            box = output[..., :4]
            confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            class_probs = 1 / (1 + np.exp(-output[..., 5:]))

            # Conversion t_x, t_y, t_w, t_h → x1, y1, x2, y2
            col = np.arange(grid_w)
            row = np.arange(grid_h)
            cx, cy = np.meshgrid(col, row)
            cx = cx[..., np.newaxis]
            cy = cy[..., np.newaxis]

            pw = self.anchors[:output.shape[2], :, 0]
            ph = self.anchors[:output.shape[2], :, 1]

            x = (1 / (1 + np.exp(-box[..., 0])) + cx) / grid_w
            y = (1 / (1 + np.exp(-box[..., 1])) + cy) / grid_h
            w = (np.exp(box[..., 2]) * pw) / self.model.input.shape[2]
            h = (np.exp(box[..., 3]) * ph) / self.model.input.shape[1]

            x1 = (x - w / 2) * image_size[1]
            y1 = (y - h / 2) * image_size[0]
            x2 = (x + w / 2) * image_size[1]
            y2 = (y + h / 2) * image_size[0]

            boxes_list.append(np.stack([x1, y1, x2, y2], axis=-1))
            box_confidences.append(confidence)
            box_class_probs.append(class_probs)

        return boxes_list, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filtre les boîtes selon le seuil de confiance"""
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

        return (np.concatenate(filtered_boxes),
                np.concatenate(box_classes),
                np.concatenate(box_scores))

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Applique la suppression non maximale"""
        from tensorflow.image import non_max_suppression

        final_boxes = []
        final_classes = []
        final_scores = []

        for cls in set(box_classes):
            cls_mask = np.where(box_classes == cls)
            cls_boxes = filtered_boxes[cls_mask]
            cls_scores = box_scores[cls_mask]
            if len(cls_boxes) == 0:
                continue

            selected_indices = non_max_suppression(cls_boxes, cls_scores,
                                    max_output_size=cls_boxes.shape[0],
                                    iou_threshold=self.nms_t)
            final_boxes.append(cls_boxes[selected_indices])
            final_classes.append(np.full(len(selected_indices), cls))
            final_scores.append(cls_scores[selected_indices])

        return (np.concatenate(final_boxes),
                np.concatenate(final_classes),
                np.concatenate(final_scores))

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """Affiche les boîtes sur l'image et sauvegarde si 's' pressé"""
        color_box = (255, 0, 0)
        color_text = (0, 0, 255)
        thickness_box = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness_text = 1
        line_type = cv2.LINE_AA

        for box, cls, score in zip(boxes, box_classes, box_scores):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), color_box, thickness_box)
            text = f"{self.class_names[cls]} {score:.2f}"
            y_text = max(y1 - 5, 0)
            cv2.putText(image, text, (x1, y_text), font, font_scale,
                        color_text, thickness_text, line_type)

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            output_dir = "detections"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, file_name)
            cv2.imwrite(output_path, image)
        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """Fait les prédictions sur toutes les images d'un dossier"""
        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)
        predictions = []

        # Prédire chaque image
        for i, img in enumerate(pimages):
            output = self.model.predict(np.expand_dims(img, axis=0))
            if not isinstance(output, list):
                output = [output]

            boxes, box_confidences, box_class_probs = self.process_outputs(output, 
                                                                image_shapes[i])
            f_boxes, b_classes, b_scores = self.filter_boxes(boxes, box_confidences,
                                                             box_class_probs)
            nms_boxes, nms_classes, nms_scores = self.non_max_suppression(f_boxes,
                                                        b_classes, b_scores)

            file_name = os.path.basename(image_paths[i])
            self.show_boxes(images[i], nms_boxes, nms_classes,
                            nms_scores, file_name)

            predictions.append((nms_boxes, nms_classes, nms_scores))

        return predictions, image_paths
