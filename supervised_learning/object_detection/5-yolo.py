#!/usr/bin/env python3
import cv2
import numpy as np
import os


class Yolo:
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialise le modèle YOLO"""
        import tensorflow as tf
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

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Affiche l'image avec les boîtes, scores et noms de classes.
        Si 's' est pressé, sauvegarde dans ./detections/
        """
        # Couleurs et styles
        color_box = (255, 0, 0)       # Bleu pour les boîtes
        color_text = (0, 0, 255)      # Rouge pour le texte
        thickness_box = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness_text = 1
        line_type = cv2.LINE_AA

        # Dessiner chaque boîte
        for box, cls, score in zip(boxes, box_classes, box_scores):
            x1, y1, x2, y2 = box.astype(int)
            # Dessiner la boîte
            cv2.rectangle(image, (x1, y1), (x2, y2), color_box, thickness_box)
            # Texte à afficher : nom + score
            text = f"{self.class_names[cls]} {score:.2f}"
            y_text = max(y1 - 5, 0)
            cv2.putText(image, text, (x1, y_text), font, font_scale,
                        color_text, thickness_text, line_type)

        # Afficher l'image
        cv2.imshow(file_name, image)
        key = cv2.waitKey(0) & 0xFF

        # Si 's', sauvegarder
        if key == ord('s'):
            output_dir = "detections"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, file_name)
            cv2.imwrite(output_path, image)

        # Fermer la fenêtre
        cv2.destroyAllWindows()
