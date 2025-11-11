# 🧠 MNIST Perturbed Agents

Ce projet implémente plusieurs **agents d’apprentissage automatique** pour résoudre la tâche de classification **MNIST Perturbed**, où les images du jeu de données MNIST sont soumises à des permutations de pixels entre les tâches.  
L’objectif est d’évaluer la robustesse et la capacité de généralisation de différents modèles sous des contraintes limitées (temps, CPU, mémoire).

---

## 📂 Structure du projet

```bash
📦 mnist_perturbed_agents/
│
├── 📁 models/
│   ├── 📁 KNN/
│   │   └── knn.py
│   │
│   ├── 📁 Linear/
│   │   └── linear.py
│   │
│   ├── 📁 Logistic_Regression/
│   │   └── logistic_regression.py
│   │
│   ├── 📁 MLP/
│   │   ├── agent_Bruce_Wayne.py
│   │   ├── agent_James_Bond.py
│   │   ├── agent_James_Bond_New_Generation_1.py
│   │   ├── agent_James_Bond_New_Generation_2.py
│   │   ├── agent_Peter_Parker.py
│   │   ├── agent_mario.py
│   │   ├── agent_mlp_v3.py
│   │   ├── mlp_v0.py
│   │   └── mlp_v1.py
│   │
│   ├── 📁 Random/
│   │   └── random.py
│   │
│   └── 📁 Xg_boost/
│       └── xg_boost.py
│
├── 📁 notebooks/
│   ├── Knn_experiment.ipynb
│   ├── Logistic_Regression_experiment.ipynb
│   ├── MLP_James_Bond_experiment.ipynb
│   ├── MLP_New_Gen_experiment.ipynb
│   ├── MLP_experiment.ipynb
│   ├── XGBoost_experiment.ipynb
│   ├── experiment0.ipynb
│   ├── grid_search_mlp.ipynb
│   └── visualization_mlp.ipynb
│
├── 📁 utils/
│   └── visualization.py
│
├── .gitignore
├── README.md
├── agent.py
├── pyproject.toml
├── report.ipynb
└── requirements.txt


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
    python3 -m venv venv

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

    

