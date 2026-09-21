# Reinforcement Learning

## Description

Ce projet est une introduction au **Reinforcement Learning (RL)**, une branche du Machine Learning dans laquelle un agent apprend à prendre des décisions en interagissant avec un environnement.

Contrairement à l'apprentissage supervisé, l'agent ne reçoit pas directement la bonne réponse. Il réalise des actions, reçoit des récompenses ou des pénalités, puis utilise ces expériences pour améliorer progressivement son comportement.

Dans ce projet, nous étudions notamment les **Markov Decision Processes (MDP)**, les **policies**, les **value functions**, le **Q-learning** et le compromis entre **exploration et exploitation**.

Le projet utilise principalement **Python**, **NumPy** et **Gymnasium**.

---

## Learning Objectives

À la fin de ce projet, l'objectif est d'être capable d'expliquer et d'utiliser les concepts suivants :

-   Qu'est-ce que le Reinforcement Learning ?
    
-   Qu'est-ce qu'un Markov Decision Process (MDP) ?
    
-   Qu'est-ce qu'un environnement ?
    
-   Qu'est-ce qu'un agent ?
    
-   Qu'est-ce qu'un état ?
    
-   Qu'est-ce qu'une action ?
    
-   Qu'est-ce qu'une récompense ?
    
-   Qu'est-ce qu'une policy function ?
    
-   Qu'est-ce qu'une value function ?
    
-   Qu'est-ce qu'une state-value function ?
    
-   Qu'est-ce qu'une action-value function ?
    
-   Qu'est-ce qu'un discount factor ?
    
-   Qu'est-ce que l'équation de Bellman ?
    
-   Qu'est-ce que l'epsilon-greedy ?
    
-   Qu'est-ce que le Q-learning ?
    
-   Comment utiliser Gymnasium pour créer et manipuler un environnement de Reinforcement Learning ?
    

---

## Concepts principaux

### Reinforcement Learning

Le Reinforcement Learning consiste à entraîner un **agent** à prendre des décisions dans un **environnement**.

À chaque étape :

```
État → Action → Récompense → Nouvel état
```

L'objectif de l'agent est de maximiser la somme des récompenses obtenues au cours du temps.

---

### Agent

L'agent est le système qui prend les décisions.

Exemple :

```
Agent : un robot
Action : avancer, reculer, tourner
```

L'agent observe son environnement et choisit une action.

---

### Environment

L'environnement représente le monde dans lequel l'agent évolue.

Il reçoit une action de l'agent et retourne notamment :

-   le nouvel état ;
    
-   une récompense ;
    
-   une information indiquant si l'épisode est terminé.
    

Avec Gymnasium, cela peut être représenté par :

```
observation, reward, terminated, truncated, info = env.step(action)
```

---

### State

Le state représente la situation actuelle de l'agent dans l'environnement.

Par exemple, dans un jeu :

```
State = position actuelle du joueur
```

Le state permet à l'agent de savoir dans quelle situation il se trouve avant de choisir une action.

---

### Action

Une action représente une décision prise par l'agent.

Par exemple :

```
0 → gauche
1 → droite
2 → avancer
3 → reculer
```

Les actions disponibles dépendent de l'environnement.

---

### Reward

La reward est la récompense reçue par l'agent après une action.

Elle permet à l'agent de savoir si son comportement est intéressant ou non.

Exemple :

```
Atteindre l'objectif → +1
Tomber dans un piège → -1
Déplacement normal → 0
```

L'agent cherche généralement à maximiser les récompenses cumulées.

---

## Policy Function

La **policy** définit la manière dont l'agent choisit une action à partir d'un état.

On peut la représenter ainsi :

```
State → Policy → Action
```

Par exemple :

```
État : obstacle devant le robot
       ↓
    Policy
       ↓
Action : tourner à gauche
```

La policy peut être déterministe ou probabiliste.

---

## Value Function

La **value function** estime la valeur d'un état en fonction des récompenses futures que l'agent peut espérer obtenir.

Elle répond essentiellement à la question :

> « Si je suis dans cet état, quelle récompense totale puis-je espérer obtenir dans le futur ? »

La value function est donc différente de la reward immédiate.

---

## State-Value Function

La **state-value function**, généralement notée `V(s)`, estime la valeur d'un état.

```
V(s)
```

Elle répond à la question :

> « Quelle est la valeur de cet état si je continue à suivre ma policy ? »

---

## Action-Value Function

L'**action-value function**, généralement notée `Q(s, a)`, estime la valeur d'une action lorsqu'elle est effectuée dans un état donné.

```
Q(s, a)
```

Elle répond à la question :

> « Quelle récompense future puis-je espérer si je suis dans l'état `s` et que je réalise l'action `a` ? »

Le Q-learning repose principalement sur cette fonction.

---

## Discount Factor

Le **discount factor**, généralement noté `γ` (gamma), permet de déterminer l'importance des récompenses futures.

Sa valeur est généralement comprise entre :

```
0 ≤ γ ≤ 1
```

Un gamma proche de `0` signifie que l'agent accorde davantage d'importance aux récompenses immédiates.

Un gamma proche de `1` signifie que l'agent accorde davantage d'importance aux récompenses futures.

---

## Bellman Equation

L'équation de Bellman permet de décomposer la valeur d'un état ou d'une action en fonction :

-   de la récompense actuelle ;
    
-   de la valeur future.
    

Pour la Q-value, la mise à jour utilisée par le Q-learning est :

```
Q(s, a) ← Q(s, a) + α [r + γ max Q(s', a') - Q(s, a)]
```

avec :

-   `s` : état actuel ;
    
-   `a` : action actuelle ;
    
-   `r` : récompense obtenue ;
    
-   `s'` : nouvel état ;
    
-   `α` : learning rate ;
    
-   `γ` : discount factor.
    

---

## Epsilon-Greedy

L'agent doit trouver un équilibre entre :

-   **exploration** : essayer de nouvelles actions ;
    
-   **exploitation** : utiliser ce qu'il a déjà appris.
    

La stratégie **epsilon-greedy** permet de gérer ce compromis.

Avec une probabilité `epsilon`, l'agent choisit une action aléatoire.

Sinon, il choisit l'action ayant la meilleure Q-value connue.

```
ε → exploration
1 - ε → exploitation
```

Cette stratégie permet à l'agent de découvrir de nouvelles possibilités tout en exploitant les connaissances déjà acquises.

---

## Q-Learning

Le **Q-learning** est un algorithme de Reinforcement Learning permettant d'apprendre une fonction `Q(s, a)`.

L'objectif est d'estimer la valeur de chaque action possible dans chaque état.

Au fur et à mesure des interactions avec l'environnement, la Q-table est mise à jour.

À terme, l'agent peut utiliser cette table pour choisir les actions qui maximisent les récompenses futures.

---

## Gymnasium

**Gymnasium** est une bibliothèque Python permettant de créer et d'utiliser des environnements de Reinforcement Learning.

Dans ce projet, elle permet notamment de travailler avec l'environnement **FrozenLake**.

Exemple de création d'un environnement :

```
import gymnasium as gym

env = gym.make("FrozenLake-v1")
```

Réinitialisation de l'environnement :

```
state, info = env.reset()
```

Exécution d'une action :

```
new_state, reward, terminated, truncated, info = env.step(action)
```

---

## FrozenLake

FrozenLake est un environnement dans lequel un agent doit se déplacer sur une grille afin d'atteindre un objectif.

L'agent doit éviter les cases dangereuses.

Un exemple de grille peut être représenté ainsi :

```
S F F F
F H F H
F F F H
H F F G
```

Avec :

```
S = Start
F = Frozen surface
H = Hole
G = Goal
```

L'objectif de l'agent est de partir de `S` et d'atteindre `G` sans tomber dans un trou.

Cet environnement permet de mettre en pratique :

-   les états ;
    
-   les actions ;
    
-   les récompenses ;
    
-   les policies ;
    
-   les value functions ;
    
-   l'exploration ;
    
-   le Q-learning.
    

---

## Requirements

Les fichiers sont interprétés et exécutés dans l'environnement suivant :

-   Ubuntu 20.04 LTS
    
-   Python 3.9
    
-   NumPy 1.25.2
    
-   Gymnasium 0.29.1
    
-   pycodestyle 2.11.1
    

---

## Installation

### 1\. Créer un environnement virtuel

Il est recommandé d'utiliser un environnement virtuel afin d'isoler les dépendances du projet.

```
python3 -m venv .venv
```

Activer l'environnement :

```
source .venv/bin/activate
```

Vous devriez voir `.venv` au début de votre terminal :

```
(.venv) user@machine:project$
```

### 2\. Installer les dépendances

Une fois l'environnement virtuel activé :

```
pip install numpy==1.25.2
pip install gymnasium==0.29.1
pip install Pillow==10.3.0
pip install h5py==3.11.0
```

Dans un environnement virtuel, il n'est pas nécessaire d'utiliser `--user`.

---

## Code Style

Le code doit respecter le standard **pycodestyle 2.11.1**.

Installation :

```
pip install pycodestyle==2.11.1
```

Vérification d'un fichier :

```
pycodestyle filename.py
```

---

## Project Requirements

Tous les fichiers Python doivent respecter les règles suivantes :

-   commencer par `#!/usr/bin/env python3` ;
    
-   se terminer par une nouvelle ligne ;
    
-   être exécutables ;
    
-   respecter le style pycodestyle ;
    
-   contenir une documentation de module ;
    
-   toutes les classes doivent avoir une documentation ;
    
-   toutes les fonctions doivent avoir une documentation ;
    
-   utiliser Python 3.9 ;
    
-   utiliser NumPy 1.25.2 ;
    
-   utiliser Gymnasium 0.29.1 ;
    
-   utiliser le minimum d'opérations nécessaire.
    

Pour rendre un fichier exécutable :

```
chmod +x filename.py
```

---

## Resources

### Reinforcement Learning

-   MIT 6.S191: Reinforcement Learning
    
-   An Introduction to Reinforcement Learning
    
-   An Introduction to Q-Learning: A Tutorial For Beginners
    
-   Q-Learning
    
-   Q-Learning Explained
    
-   Exploration vs. Exploitation
    

### Markov Decision Processes

-   Markov Decision Processes (MDPs)
    
-   Markov Decision Processes
    
-   Expected Return
    
-   Policies and Value Functions
    
-   Optimal Policies
    

### Gymnasium

-   Gymnasium
    
-   Gymnasium FrozenLake environment
    
-   Gymnasium Q-Learning with FrozenLake
    
-   Gymnasium Environment API
    

---

## Project Structure

La structure du projet peut être organisée de la manière suivante :

```
holbertonschool-machine_learning/
│
├── README.md
│
├── 0-load_env.py
├── 1-q_init.py
├── 2-epsilon_greedy.py
├── ...
│
└── ...
```

Les noms exacts et le contenu des fichiers dépendent des tâches fournies dans le projet.

---

## Goal of the Project

L'objectif final est de comprendre comment un agent peut **apprendre à prendre des décisions par essais et erreurs**.

Le projet permet de passer progressivement de la théorie :

```
MDP
 ↓
State
 ↓
Action
 ↓
Reward
 ↓
Policy / Value Function
 ↓
Q-Learning
 ↓
Agent capable d'apprendre
```

à une implémentation concrète avec **Python, NumPy et Gymnasium**.