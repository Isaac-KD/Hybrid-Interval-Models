from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go

def generate_data(
    macsum, 
    phi_true,
    n_samples: int, 
    noise_level: float = 0.01,
    borne: int = 50.0,
    generation: str = "uniform",
    multi: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Génère un jeu de données (X, Y) qui suit la logique du modèle Macsum.
    """
    n = len(phi_true)
    X=np.empty((0, n))
    if generation == "uniform":
        X = np.random.uniform(low=-borne, high=borne, size=(n_samples, n))
        
    elif generation == "gauss":
        for i in range(multi):
             X = np.concatenate((np.random.normal(loc= borne*i*np.sqrt(np.pi), scale=borne / 3, size=(n_samples, n)),X), axis=0)
    
    Y = np.zeros(n_samples*multi)

    for i in range(n_samples*multi):
        x = X[i]

        y_lower_true, y_upper_true = macsum.prediction(x)
        
        
        y_inside = (y_upper_true + y_lower_true)/2
        
        # 3. Ajouter un peu de bruit de mesure (gaussien) pour plus de réalisme.
        interval_spread = y_upper_true - y_lower_true
        if interval_spread < 1e-6:
             interval_spread = 1.0 # Éviter la division par zéro
             
        noise = np.random.randn() * noise_level * interval_spread
        
        Y[i] = y_inside + noise

    return X, Y

def save_data(X,Y,name="data"):

    # Convertir la matrice numpy en DataFrame
    label = [ f"x{i}"for i in range( X.shape[1])]
    label.insert(-1,"target")
    df = pd.DataFrame(np.column_stack((X, Y)), columns=label)

    # Sauvegarder dans CSV
    df.to_csv(name+'.csv', index=False)

def plot_points_2d(points, group_size, show_labels=False):
    """
    Affiche des groupes de points 2D sur un repère cartésien.
    Chaque groupe de taille n est affiché avec une couleur différente.

    Args:
        points (list of list or tuples): Liste de points [x, y].
        group_size (int): Taille de chaque groupe.
        show_labels (bool): Afficher les coordonnées à côté des points.
    """
    if group_size <= 0:
        raise ValueError("group_size must be a positive integer")

    x = [p[0] for p in points]
    y = [p[1] for p in points]

    total_points = len(points) 
    num_groups = total_points // group_size

    plt.figure(figsize=(6, 6))
    cmap = plt.get_cmap('tab10')

    for i in range(num_groups):
        start = i * group_size
        end = start + group_size
        plt.scatter(x[start:end], y[start:end], color=cmap(i % 10), s=50, label=f'Groupe {i+1}')
    
    # Plot remaining points if any
    remainder = total_points % group_size
    if remainder:
        start = num_groups * group_size
        plt.scatter(x[start:], y[start:], color=cmap(num_groups % 10), s=50, label=f'Groupe {num_groups+1}')

    if show_labels:
        for i, (xi, yi) in enumerate(points):
            plt.text(xi + 0.1, yi + 0.1, f'({xi:.2f}, {yi:.2f})', fontsize=9)

    plt.axhline(0, color='gray', linewidth=1)
    plt.axvline(0, color='gray', linewidth=1)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Points dans le plan 2D par groupe')
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()

def plot_3d_points(X, Y):
    scatter = go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=Y,
        mode='markers',
        marker=dict(
            size=5,
            color=Y,
            colorscale='Viridis',
            colorbar=dict(title='Valeur Y'),
            opacity=0.8
        )
    )

    layout = go.Layout(
        scene=dict(
            xaxis_title='X1',
            yaxis_title='X2',
            zaxis_title='Y'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()