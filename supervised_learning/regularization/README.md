📌 Fiche 1 — Regularization (Régularisation)

🔹 Définition
La régularisation est un ensemble de techniques utilisées pour réduire le surapprentissage (overfitting) d’un modèle.
Elle agit en contraignant la complexité du modèle (réduire ses degrés de liberté, sa capacité à mémoriser les données d’entraînement au lieu de généraliser).

🔹 But
Améliorer la généralisation : bonnes performances non seulement sur l’entraînement, mais aussi sur la validation/test.
Empêcher le modèle d’apprendre trop de bruit ou de particularités spécifiques aux données d’entraînement.

📌 Fiche 2 — L1 et L2 Regularization

🔹 Principe général
On ajoute une pénalité sur les poids ww du modèle dans la fonction de coût :

Lossreg=Lossoriginal+λ⋅Penalty(w)

🔹 L1 (Lasso regularization)
Pénalité :

PenaltyL1=∑∣wi∣

Effet :
Encourage les poids à devenir exactement 0 → favorise la sparsité (sélection de variables).
Utile pour réduire la dimensionnalité et faire du feature selection.

🔹 L2 (Ridge regularization)
Pénalité : PenaltyL2=∑wi2
Effet :
Réduit l’amplitude des poids mais rarement à 0.
Encourage des poids petits et répartis.
Plus stable numériquement que L1.

🔹 Différences clés
L1 : produit des poids exactement nuls → sélection de variables.
L2 : réduit les poids mais les garde non nuls → meilleure stabilité.
Elastic Net : combinaison des deux.

📌 Fiche 3 — Dropout

🔹 Définition
Technique de régularisation pour réseaux de neurones.
Pendant l’entraînement, on désactive aléatoirement un certain pourcentage de neurones (ex. 50%).

🔹 But
Évite que le réseau devienne trop dépendant d’un petit ensemble de neurones.
Force le modèle à apprendre des représentations redondantes et robustes.

🔹 Remarque
Utilisé seulement à l’entraînement.
En test/inférence, on garde tous les neurones mais on réduit leurs poids par le taux de dropout.

📌 Fiche 4 — Early Stopping

🔹 Définition
Stratégie qui consiste à arrêter l’entraînement automatiquement quand la performance sur les données de validation commence à se dégrader.

🔹 But
Éviter que le modèle continue à apprendre le bruit (overfitting).

🔹 Exemple
On suit la val_loss (erreur sur la validation).
Si elle n’améliore plus après N itérations → on stoppe l’entraînement.

📌 Fiche 5 — Data Augmentation

🔹 Définition
Technique pour créer artificiellement plus de données d’entraînement à partir des données existantes.

🔹 Exemples en vision
Rotation, recadrage, zoom, bruit, inversion horizontale, changement de luminosité.

🔹 Exemples en NLP
Synonym replacement, back-translation.

🔹 But
Réduire le surapprentissage.
Améliorer la robustesse.



📌 Fiche 6 — Implémentation (Numpy vs TensorFlow)

🔹 L1 et L2
Numpy :
import numpy as np

def l1_penalty(w, lambd=0.01):
    return lambd * np.sum(np.abs(w))

def l2_penalty(w, lambd=0.01):
    return lambd * np.sum(w**2)

TensorFlow/Keras :
from tensorflow.keras import regularizers

# L1
Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01))

# L2
Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))

# L1 + L2 (Elastic Net)
Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(0.01, 0.01))


🔹 Dropout
Numpy (simulé) :
def dropout_layer(X, drop_rate=0.5):
    mask = (np.random.rand(*X.shape) > drop_rate).astype(float)
    return (X * mask) / (1 - drop_rate)

TensorFlow/Keras :
from tensorflow.keras.layers import Dropout

model.add(Dropout(0.5))


🔹 Early Stopping
Numpy : implémentation maison → surveiller la val_loss, arrêter si elle n’améliore pas.
TensorFlow/Keras :
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

🔹 Data Augmentation
Numpy : écrit à la main (rotations, bruit, flips, etc.).
TensorFlow/Keras (vision) :
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
