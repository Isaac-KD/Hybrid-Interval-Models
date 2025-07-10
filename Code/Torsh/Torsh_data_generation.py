from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import torch 
from Code.Torsh.TorchMacsumAggregationLearning import *

def generate_data(
    macsum_model: 'Macsum', 
    phi_true_np: np.ndarray,
    n_samples: int,
    noise_level: float = 0.01,
    borne: float = 50.0, 
    generation: str = "uniform",
    multi: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Génère un jeu de données (X, Y) qui suit la logique du modèle Macsum (version PyTorch).
    Utilise le dtype du paramètre _phi du modèle pour la création des tenseurs.
    """
    n_features = len(phi_true_np) # 
    X_list = []

    if generation == "uniform":
        X_gen = np.random.uniform(low=-borne, high=borne, size=(n_samples, n_features))
        X_list.append(X_gen)
    elif generation == "gauss":
        samples_per_multi = n_samples // multi if multi > 0 else n_samples
        remaining_samples = n_samples % multi if multi > 0 else 0
        
        for i in range(multi):
            current_samples = samples_per_multi + (1 if i < remaining_samples else 0)
            if current_samples == 0:
                continue
            mean_offset = borne * i * np.sqrt(np.pi) 
            X_gen = np.random.normal(loc=mean_offset, scale=borne / 3, size=(current_samples, n_features))
            X_list.append(X_gen)
    else:
        raise ValueError(f"Unknown generation type: {generation}")

    if not X_list: # Si aucun échantillon n'a été généré (par ex. n_samples=0 ou multi incorrect)
        return np.array([]).reshape(0, n_features), np.array([])

    X = np.concatenate(X_list, axis=0)
    
    current_total_samples = X.shape[0]

    # Récupérer le dtype et device directement depuis le modèle
    model_device = macsum_model._phi.device
    model_dtype = macsum_model._phi.dtype 

    # Mettre le modèle en mode évaluation (déjà fait implicitement si on n'appelle pas .train())
    macsum_model.eval()
    # Le modèle devrait déjà être sur le bon device, mais on peut le réassurer
    macsum_model.to(model_device) 
    
    # Convertir phi_true_np au bon type numpy correspondant au model_dtype
    if model_dtype == torch.float32:
        phi_true_np_casted = phi_true_np.astype(np.float32)
    elif model_dtype == torch.float64:
        phi_true_np_casted = phi_true_np.astype(np.float64)
    else:
        # Gérer d'autres dtypes si nécessaire, ou lever une erreur
        phi_true_np_casted = phi_true_np # Fallback, pourrait causer des problèmes

    with torch.no_grad(): # Important si on modifie les paramètres du modèle
        # Créer le tenseur avec le dtype et device du modèle
        new_phi_tensor = torch.tensor(phi_true_np_casted, dtype=model_dtype, device=model_device)
        macsum_model._phi.data.copy_(new_phi_tensor) 
        macsum_model._recompute_phi_derived_attrs()

    # Convertir X au bon type numpy et ensuite au tenseur avec le model_dtype
    if model_dtype == torch.float32:
        X_np_casted = X.astype(np.float32)
    elif model_dtype == torch.float64:
        X_np_casted = X.astype(np.float64)
    else:
        X_np_casted = X # Fallback

    X_tensor = torch.tensor(X_np_casted, dtype=model_dtype, device=model_device)

    with torch.no_grad(): # Pas besoin de gradients pour la génération de données
        y_lower_true_batch, y_upper_true_batch = macsum_model(X_tensor)

    y_lower_true_np = y_lower_true_batch.cpu().numpy()
    y_upper_true_np = y_upper_true_batch.cpu().numpy()
    
    # Initialiser Y avec le dtype approprié (correspondant aux prédictions)
    Y = np.zeros(current_total_samples, dtype=y_lower_true_np.dtype)


    for i in range(current_total_samples):
        y_l_true = y_lower_true_np[i]
        y_u_true = y_upper_true_np[i]
        
        y_inside = (y_u_true + y_l_true) / 2.0
        
        interval_spread = y_u_true - y_l_true
        if abs(interval_spread) < 1e-9: # Utiliser abs et une petite tolérance
             # Si l'intervalle est quasi nul, le bruit sera basé sur noise_level absolu * une échelle
             # ou une valeur par défaut pour éviter un bruit nul ou infini si noise_level est grand.
             # On peut aussi choisir de ne pas ajouter de bruit dans ce cas.
             noise_scale = 0.01 # Échelle par défaut pour le bruit si l'intervalle est nul
        else:
            noise_scale = interval_spread
             
        noise = np.random.randn() * noise_level * noise_scale
        
        Y[i] = y_inside + noise

    return X, Y

def save_data(X: np.ndarray, Y: np.ndarray, name: str ="data"):
    """
    Sauvegarde X et Y dans un fichier CSV.
    """
    if X.ndim != 2:
        raise ValueError("X doit être une matrice 2D.")
    if Y.ndim != 1:
        raise ValueError("Y doit être un vecteur 1D.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X et Y doivent avoir le même nombre d'échantillons.")

    labels = [f"x{i}" for i in range(X.shape[1])]
    labels.append("target") 

    # Concaténer Y comme une nouvelle colonne à X
    data_to_save = np.column_stack((X, Y))
    df = pd.DataFrame(data_to_save, columns=labels)

    df.to_csv(name + '.csv', index=False)
    print(f"Data saved to {name}.csv")


def generate_friedman1_data(n_samples: int, n_features: int, noise_std: float = 0.1):
    """
    Génère des données basées sur la fonction de test de Friedman #1.
    
    Args:
        n_samples: Nombre d'échantillons à générer.
        n_features: Nombre total de features. Doit être >= 5.
        noise_std: Écart-type du bruit gaussien ajouté à la sortie.
                   Mettre à 0 pour avoir des données parfaites (sans bruit).
    
    Returns:
        Tuple (X, y) où X.shape=(n_samples, n_features) et y.shape=(n_samples,)
    """
    if n_features < 5:
        raise ValueError("n_features must be at least 5 for Friedman #1 function.")
        
    X = np.random.rand(n_samples, n_features) * 10
    
    # Calculer la sortie Y en utilisant la formule
    # Seules les 5 premières colonnes de X sont utilisées
    term1 = 10 * np.sin(np.pi * X[:, 0] * X[:, 1])
    term2 = 20 * (X[:, 2] - 0.5)**2
    term3 = 10 * X[:, 3]
    term4 = 5 * X[:, 4]
    
    y_true = term1 + term2 + term3 + term4
    
    # Ajouter du bruit gaussien si nécessaire
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, n_samples)
        y_noisy = y_true + noise
        return X, y_noisy
    else:
        return X, y_true
    
def plot_points_2d(points, group_size, show_labels=False):
    """
    Affiche des groupes de points 2D sur un repère cartésien.
    Chaque groupe de taille n est affiché avec une couleur différente.

    Args:
        points (list of list or tuples): Liste de points [x, y].
        group_size (int): Taille de chaque groupe.
        show_labels (bool): Afficher les coordonnées à côté des points.
    """
    if not isinstance(points, (list, np.ndarray)):
        raise TypeError("points should be a list or numpy array of [x,y] coordinates")
    if len(points) == 0:
        print("No points to plot.")
        return
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Each point should be a list/tuple of 2 coordinates [x,y]")


    if group_size <= 0:
        raise ValueError("group_size must be a positive integer")

    x_coords = points[:, 0]
    y_coords = points[:, 1]

    total_points = len(points) 
    num_groups = total_points // group_size

    plt.figure(figsize=(8, 8)) # Augmenté un peu la taille
    cmap = plt.get_cmap('tab10') # Bon pour <= 10 groupes distincts

    for i in range(num_groups):
        start = i * group_size
        end = start + group_size
        plt.scatter(x_coords[start:end], y_coords[start:end], color=cmap(i % 10), s=50, label=f'Groupe {i+1}')
    
    remainder = total_points % group_size
    if remainder > 0 : # Vérifier si remainder > 0 pour éviter un label inutile
        start = num_groups * group_size
        plt.scatter(x_coords[start:], y_coords[start:], color=cmap(num_groups % 10), s=50, label=f'Groupe {num_groups+1} (partiel)')

    if show_labels:
        for i, (xi, yi) in enumerate(points):
            plt.text(xi + (plt.xlim()[1] - plt.xlim()[0])*0.01, # Petit offset relatif à la taille du graphe
                     yi + (plt.ylim()[1] - plt.ylim()[0])*0.01, 
                     f'({xi:.2f}, {yi:.2f})', fontsize=8) # Taille de police réduite

    plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Points dans le plan 2D par groupe')
    plt.gca().set_aspect('equal', adjustable='box')
    if num_groups > 0 or remainder > 0 : # Afficher la légende seulement s'il y a des points
        plt.legend()
    plt.show()

def plot_3d_points(X: np.ndarray, Y: np.ndarray):
    """
    Affiche les points (X[:,0], X[:,1], Y) en 3D.
    """
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("X doit être une matrice 2D avec au moins 2 colonnes.")
    if Y.ndim != 1:
        raise ValueError("Y doit être un vecteur 1D.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X et Y doivent avoir le même nombre d'échantillons.")

    scatter = go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=Y,
        mode='markers',
        marker=dict(
            size=5,
            color=Y,
            colorscale='Viridis', # 'Plasma', 'Inferno', 'Magma' sont aussi possible
            colorbar=dict(title='Valeur Y'),
            opacity=0.8
        )
    )

    layout = go.Layout(
        title='Visualisation 3D des données (X1, X2, Y)', 
        scene=dict(
            xaxis_title='X1 (Première caractéristique)',
            yaxis_title='X2 (Deuxième caractéristique)',
            zaxis_title='Y (Cible)'
        ),
        margin=dict(l=0, r=0, b=0, t=40) 
    )

    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()
