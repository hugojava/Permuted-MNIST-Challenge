# 🧠 MNIST Perturbed Agents

Ce projet implémente plusieurs **agents d’apprentissage automatique** pour résoudre la tâche de classification **MNIST Perturbed**, où les images du jeu de données MNIST sont soumises à des permutations de pixels entre les tâches.  
L’objectif est d’évaluer la robustesse et la capacité de généralisation de différents modèles sous des contraintes limitées (temps, CPU, mémoire).

---

## 📂 Structure du projet



```bash
permuted_mnist/
├── permuted_mnist/
│   ├── agent/
│   │   ├── linear/
│   │   │   └── agent.py
│   │   ├── random/
│   │   │   └── agent.py
│   │   └── torch_mlp/
│   │       └── agent.py
│   │
│   ├── data/
│   │   ├── mnist_test_images.npy
│   │   ├── mnist_test_labels.npy
│   │   ├── mnist_train_images.npy
│   │   └── mnist_train_labels.npy
│   │
│   ├── env/
│   │   ├── __init__.py
│   │   ├── permuted_mnist.py
│   │   └── renderer.py
│   │
│   └── __init__.py
│
├── permuted_mnist.egg-info/
│
├── Permuted-MNIST-*/           # Dossier contenant les données brutes et modèles
│   ├── data/MNIST/raw/
│   └── models/
│       ├── KNN/
│       │   └── knn.py
│       ├── Linear/
│       │   └── linear.py
│       ├── Logistic_Regression/
│       │   └── logistic_regression.py
│       ├── MLP/
│       │   ├── agent_Bruce_Wayne.py
│       │   ├── agent_James_Bond.py
│       │   ├── agent_James_Bond_New_Generation_1.py
│       │   ├── agent_James_Bond_New_Generation_2.py
│       │   ├── agent_mario.py
│       │   ├── agent_Peter_Parker.py
│       │   ├── mlp_v0.py
│       │   ├── mlp_v1.py
│       │   └── mlp_v3.py
│       ├── Random/
│       │   └── random.py
│       └── Xg_boost/
│           └── xg_boost.py
│
├── notebooks/
│   ├── experiments/
│   │   ├── experiment0.ipynb
│   │   ├── experiment1.ipynb
│   │   ├── experiment2.ipynb
│   │   ├── experiment3.ipynb
│   │   ├── experiment4.ipynb
│   │   ├── experiment5.ipynb
│   │   ├── experiment6.ipynb
│   │   ├── grid_search_mlp.ipynb
│   │   └── visualization_mlp.ipynb
│   ├── report.ipynb
│   └── getting_started.ipynb
│
├── utils/
│   └── visualization.py
│
├── tools/
│
├── agent.py
├── README.md
├── requirements.txt
├── setup.py
├── pyproject.toml
└── .gitignore

```
---

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

    

