import matplotlib.pyplot as plt
import numpy as np

def Comparison_plots(*args):
    """
    Compare plusieurs modèles en affichant deux graphiques côte à côte :
    - Accuracy par tâche (zoom 0.9-1.0)
    - Temps d'exécution (entraînement + prédiction) par tâche
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    num_models = len(args)
    colors = plt.cm.tab10(np.linspace(0, 1, num_models))  # palette auto

    # --- Plot accuracies ---
    for (accs, times, name), color in zip(args, colors):
        tasks = np.arange(1, len(accs) + 1)
        ax1.plot(tasks, accs, 'o-', label=name, alpha=0.8, linewidth=2, color=color)

    ax1.set_xlabel('Task Number')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy per Task')
    ax1.set_ylim([0.9, 1.0])   # <-- zoom sur la plage souhaitée
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --- Plot times ---
    bar_width = 0.8 / num_models
    for i, ((accs, times, name), color) in enumerate(zip(args, colors)):
        tasks = np.arange(1, len(times) + 1)
        offsets = (i - num_models/2) * bar_width + bar_width/2
        ax2.bar(tasks + offsets, times, bar_width, label=name, alpha=0.8, color=color)

    ax2.axhline(y=60, color='red', linestyle='--', alpha=0.5, label='1 minute threshold')
    ax2.set_xlabel('Task Number')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Training + Prediction Time per Task')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()