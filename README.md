# 🧠 MNIST Perturbed Agents

Ce projet implémente plusieurs **agents d’apprentissage automatique** pour résoudre la tâche de classification **MNIST Perturbed**, où les images du jeu de données MNIST sont soumises à des permutations de pixels entre les tâches.  
L’objectif est d’évaluer la robustesse et la capacité de généralisation de différents modèles sous des contraintes limitées (temps, CPU, mémoire).

---

## 📂 Structure du projet

mettre l'arborescence

## ⚙️ Installation

1. **Cloner le dépôt :**
    ```bash
    # Cloner le répertoire permuted_mnist
    git clone https://github.com/ml-arena/permuted_mnist/
    cd permuted_mnist

    # Installer le package
    pip install -e .

    # Cloner notre package
    git clone https://github.com/hugojava/Permuted-MNIST-Challenge
    cd Permuted-MNIST-Challenge

2. **installer les dépendances :**
    ```bash
    # 1️⃣ Créer un environnement virtuel
    python -m venv venv

    # 2️⃣ L'activer
    # Sous Linux / macOS :
    source venv/bin/activate

    # Sous Windows :
    venv\Scripts\activate

    # 3️⃣ Installer les dépendances du projet
    pip install -r requirements.txt

## ⏱️ Contraintes

    Temps max par épisode : 1 minute

    Mémoire max : 4 GB

    CPU : 2 cœurs

    Pas de GPU

Les agents sont conçus pour s’exécuter efficacement dans ces conditions.

## 🧑‍💻 Auteurs

Projet développé par Hugo Bouton et Erwan Ouabdesselam, dans le cadre du challenge Permuted MNIST sur ML Arena.

    

