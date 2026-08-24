#!/usr/bin/env python3
"""Hyperparameter Optimization using GPyOpt and TensorFlow/Keras."""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.datasets import mnist
import GPyOpt

# 1. Chargement et préparation des données
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train / 255.0).astype(np.float32)
x_test = (x_test / 255.0).astype(np.float32)

# Réduction de la taille pour un entraînement rapide lors de l'optimisation
x_train, y_train = x_train[:10000], y_train[:10000]

# 2. Définition du domaine des hyperparamètres (5 hyperparamètres)
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
    {'name': 'num_units', 'type': 'discrete', 'domain': (32, 64, 128, 256)},
    {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.1, 0.5)},
    {'name': 'l2_reg', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128)}
]


def build_and_train_model(x):
    """Fonction objectif à minimiser par GPyOpt.
    
    x est un tableau 2D de forme (1, num_hyperparameters).
    """
    lr = float(x[0, 0])
    num_units = int(x[0, 1])
    dropout = float(x[0, 2])
    l2_weight = float(x[0, 3])
    batch_sz = int(x[0, 4])

    print(f"\n--- Evaluation : lr={lr:.5f}, units={num_units}, "
          f"dropout={dropout:.2f}, l2={l2_weight:.5f}, batch={batch_sz} ---")

    # Architecture du modèle
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(
            num_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_weight)
        ),
        layers.Dropout(dropout),
        layers.Dense(10, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Nom du checkpoint basé sur les valeurs des hyperparamètres
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    chkpt_filename = (
        f"model_lr{lr:.5f}_units{num_units}_drop{dropout:.2f}_"
        f"l2{l2_weight:.5f}_bs{batch_sz}.h5"
    )
    chkpt_path = os.path.join(checkpoint_dir, chkpt_filename)

    # Callbacks: Early Stopping + Checkpoint de la meilleure itération
    early_stop = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        mode='max',
        restore_best_weights=True
    )
    checkpoint = callbacks.ModelCheckpoint(
        filepath=chkpt_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=0
    )

    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=15,
        batch_size=batch_sz,
        callbacks=[early_stop, checkpoint],
        verbose=0
    )

    # Métrique satisfaisante : Précision maximale de validation
    best_val_acc = max(history.history['val_accuracy'])
    print(f"Meilleure validation accuracy: {best_val_acc:.4f}")

    # GPyOpt effectue une MINIMISATION, on retourne donc -val_accuracy
    return -best_val_acc


# 3. Lancement de l'Optimisation Bayésienne avec GPyOpt
print("Démarrage de l'optimisation bayésienne...")
optimizer = GPyOpt.methods.BayesianOptimization(
    f=build_and_train_model,
    domain=bounds,
    acquisition_type='EI',  # Expected Improvement
    exact_feval=True
)

# Nombre maximal d'itérations = 30
max_iter = 30
optimizer.run_optimization(max_iter=max_iter)

# 4. Tracé et sauvegarde du graphe de convergence
optimizer.plot_convergence('convergence_plot.png')
plt.close()

# 5. Extraction des résultats et écriture du rapport bayes_opt.txt
best_x = optimizer.x_opt
best_opt_val = -optimizer.fx_opt

report_content = f"""BAYESIAN OPTIMIZATION REPORT
============================
Total Iterations: {len(optimizer.Y)}
Best Satisficing Metric (Validation Accuracy): {best_opt_val:.4f}

Optimal Hyperparameters Found:
------------------------------
Learning Rate : {best_x[0]:.6f}
Number of Units: {int(best_x[1])}
Dropout Rate  : {best_x[2]:.4f}
L2 Weight     : {best_x[3]:.6f}
Batch Size    : {int(best_x[4])}
"""

with open('bayes_opt.txt', 'w') as f:
    f.write(report_content)

print("\nOptimisation terminée avec succès !")
print(report_content)