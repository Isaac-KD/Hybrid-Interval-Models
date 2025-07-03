import torch
from Code.Torsh.TorchMacsumAggregationLearning import *
import torch.nn.functional as F # Assurez-vous que cet import est en haut du fichier

class MacsumLayer(nn.Module):
    """
    Représente une couche de N_out unités Macsum fonctionnant en parallèle.
    Chaque unité prend un vecteur de taille N_in et produit un scalaire.
    La couche transforme donc un intervalle de vecteurs [x_lower, x_upper] 
    de taille N_in en un intervalle de vecteurs de taille N_out.
    """
    def __init__(self, in_features: int, out_features: int, 
                 model_class: Type[Macsum] = MacsumSigmoidTorch, 
                 **model_params):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Crée une liste de 'out_features' unités Macsum.
        # Chaque unité est un expert qui observe l'entrée de 'in_features' dimensions.
        self.macsum_units = nn.ModuleList([
            model_class(N=in_features, **model_params) 
            for _ in range(out_features)
        ])

    def forward(self, x_interval: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Passe avant de la couche.
        Args:
            x_interval: Un tuple (x_lower, x_upper), où chaque tenseur a une 
                        shape (batch_size, in_features).
        Returns:
            Un tuple (y_lower_vec, y_upper_vec), où chaque tenseur a une 
            shape (batch_size, out_features).
        """
        x_lower, x_upper = x_interval
        
        # Les fonctions y_lower et y_upper de Macsum sont monotones croissantes.
        # Pour obtenir l'intervalle de sortie le plus serré, on calcule :
        # - La borne inférieure de la sortie en utilisant la borne inférieure de l'entrée.
        # - La borne supérieure de la sortie en utilisant la borne supérieure de l'entrée.
        
        # Calcule les bornes pour chaque unité Macsum de la couche
        all_y_lower = []
        all_y_upper = []

        for unit in self.macsum_units:
            # y_lower_pred pour x_lower donne la borne inf de la sortie de cette unité
            y_l_unit, _ = unit(x_lower)
            all_y_lower.append(y_l_unit)
            
            # y_upper_pred pour x_upper donne la borne sup de la sortie de cette unité
            _, y_u_unit = unit(x_upper)
            all_y_upper.append(y_u_unit)

        # Empile les résultats pour former des vecteurs de sortie
        # Shape: (batch_size, out_features)
        y_lower_vec = torch.stack(all_y_lower, dim=1)
        y_upper_vec = torch.stack(all_y_upper, dim=1)
        
        return y_lower_vec, y_upper_vec

    def _recompute_phi_derived_attrs_for_all(self):
        """Appelle _recompute_phi_derived_attrs sur chaque unité de la couche."""
        for unit in self.macsum_units:
            unit._recompute_phi_derived_attrs()


class MacsumNet(MacsumSigmoidTorch):
    """
    Un réseau de neurones composé de couches de Macsum.
    Hérite de MacsumSigmoidTorch pour réutiliser les hyperparamètres de la 
    loss (alpha, gamma, k_sigmoid) et sa fonction de coût.
    """
    def __init__(self, layer_config: List[int],
                 model_class: Type[Macsum] = MacsumSigmoidTorch, 
                 **model_params):
        """
        Args:
            layer_config: Liste d'entiers définissant l'architecture.
                          Ex: [10, 20, 1] pour un réseau avec une entrée de 10,
                          une couche cachée de 20 neurones, et une sortie de 1.
            model_class: La classe de base pour les unités (ex: MacsumSigmoidTorch).
            model_params: Hyperparamètres pour les unités (alpha, gamma, k_sigmoid, etc.).
        """
        # Initialise la classe parente avec N=0 (non utilisé) mais garde les hyperparams
        # C'est cette ligne qui causait l'erreur indirectement.
        super().__init__(N=0, **model_params)
        
        if len(layer_config) < 2:
            raise ValueError("layer_config doit contenir au moins une couche d'entrée et une de sortie.")
        
        self.layer_config = layer_config
        self.layers = nn.ModuleList()

        for i in range(len(layer_config) - 1):
            in_features = layer_config[i]
            out_features = layer_config[i+1]
            # On passe les mêmes hyperparamètres à chaque unité du réseau
            layer = MacsumLayer(in_features, out_features, model_class, **model_params)
            self.layers.append(layer)

    @override
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Passe avant pour le réseau complet.
        Args:
            x: Tenseur d'entrée de shape (batch_size, N_input), où 
               N_input = self.layer_config[0].
        Returns:
            Un tuple (y_lower_final, y_upper_final), où chaque tenseur a une 
            shape (batch_size, N_output) et N_output = self.layer_config[-1].
        """
        # Pour la première couche, l'intervalle d'entrée est [x, x]
        x_interval = (x, x)
        
        # Propagation à travers les couches
        for layer in self.layers:
            x_interval = layer(x_interval)
            
        y_lower_final, y_upper_final = x_interval
        
        # Si la sortie est un scalaire par échantillon, on squeeze la dernière dimension
        if y_lower_final.shape[1] == 1:
            y_lower_final = y_lower_final.squeeze(1)
            y_upper_final = y_upper_final.squeeze(1)
            
        return y_lower_final, y_upper_final
    
    @override
    def _recompute_phi_derived_attrs(self):
        """
        CORRECTION: Ajout d'une protection.
        Cette méthode est appelée par le constructeur parent AVANT que self.layers
        ne soit défini. On vérifie donc son existence avant de l'utiliser.
        """
        if hasattr(self, 'layers'):
            for layer in self.layers:
                layer._recompute_phi_derived_attrs_for_all()

    # CORRECTION : Ajout de beta1 et beta2 et utilisation dans l'optimiseur.
    def fit_autograd(self, X_train: np.ndarray, Y_train: np.ndarray,
                     X_eval: np.ndarray = None, Y_eval: np.ndarray = None,
                     learning_rate: float = 0.001, 
                     n_epochs: int = 100, 
                     batch_size: int = 32,
                     beta1: float = 0.9, 
                     beta2: float = 0.999,
                     print_every: int = 10,
                     weight_decay: float = 1e-4,
                     epsilon_conv: float = 1e-6):
        """
        Entraînement du réseau MacsumNet avec PyTorch Autograd.
        Cette méthode est plus adaptée aux réseaux profonds que la méthode avec 
        gradient manuel. L'approximation (traiter les permutations comme fixes
        pour le calcul du gradient) est la même.
        """
        X_train_t = torch.tensor(X_train, dtype=torch.float64, device=DEVICE)
        Y_train_t = torch.tensor(Y_train, dtype=torch.float64, device=DEVICE)
        
        train_dataset = TensorDataset(X_train_t, Y_train_t)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        self.to(DEVICE)
        self.train() 

        # L'optimizer collecte tous les paramètres _phi de toutes les unités
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay = weight_decay)
        
        self.history = []
        print(f"Starting PyTorch Autograd training for MacsumNet...")

        for epoch in trange(1, n_epochs + 1, desc="Autograd Training (MacsumNet)"):
            epoch_total_loss_sum = 0.0
            epoch_total_samples = 0
            
            for X_batch, Y_batch in train_dataloader:
                optimizer.zero_grad()
                
                # 1. Forward pass
                y_lower_pred, y_upper_pred = self(X_batch)
                
                # 2. Calculer la loss (utilise la fonction de coût de la classe parente)
                loss_per_sample = self._loss(Y_batch, (y_lower_pred, y_upper_pred))
                
                # 3. Calculer la loss moyenne du batch et faire la passe backward
                mean_loss = loss_per_sample.mean()
                mean_loss.backward()
                
                # 4. Mettre à jour les poids (les _phi de tout le réseau)
                optimizer.step()
                
                # 5. Recalculer les permutations pour la prochaine itération
                self._recompute_phi_derived_attrs()

                epoch_total_loss_sum += loss_per_sample.sum().item()
                epoch_total_samples += X_batch.shape[0]

            # Évaluation et logs de fin d'époque
            avg_epoch_loss_train = epoch_total_loss_sum / epoch_total_samples if epoch_total_samples > 0 else float('nan')
            if print_every > 0 and (epoch % print_every == 0 or epoch == n_epochs):
                if X_eval is not None and Y_eval is not None:
                    val_metrics = evaluate_model_complet(
                        X_eval, Y_eval, self, avg_epoch_loss_train, batch_size=batch_size
                    )
                    val_metrics['epoch'] = epoch
                    self.history.append(val_metrics)
        
        print("Training finished.")
        return self

class MacsumNetWithActivation(MacsumNet): # Hérite de MacsumNet
    """
    Une version de MacsumNet qui ajoute une fonction d'activation
    non-linéaire (ex: ReLU) entre les couches.
    """
    def __init__(self, layer_config: List[int],
                 model_class: Type[Macsum] = MacsumSigmoidTorch,
                 activation_fn = F.relu, 
                 **model_params):
        
        # Initialise la structure de base (couches, etc.) via le constructeur parent
        super().__init__(layer_config, model_class, **model_params)
        
        self.activation_fn = activation_fn
        # On utilise tqdm.write si on est dans une boucle tqdm, sinon print
        try:
            from tqdm import tqdm
            tqdm.write(f"Création d'un MacsumNet avec la fonction d'activation: {activation_fn.__name__ if activation_fn else 'Aucune'}")
        except (ImportError, AttributeError):
             print(f"Création d'un MacsumNet avec la fonction d'activation: {activation_fn.__name__ if activation_fn else 'Aucune'}")


    @override
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Passe avant pour le réseau complet AVEC fonctions d'activation.
        """
        x_interval = (x, x)
        
        num_layers = len(self.layers)
        # On itère sur toutes les couches
        for i, layer in enumerate(self.layers):
            # 1. Passe à travers la couche Macsum
            x_interval = layer(x_interval)
            
            # 2. Appliquer la fonction d'activation si ce N'EST PAS la dernière couche
            if self.activation_fn is not None and i < num_layers - 1:
                x_lower, x_upper = x_interval
                
                # La fonction d'activation est appliquée sur les deux bornes.
                # Comme ReLU(x) = max(0,x) est une fonction croissante, on a
                # ReLU([a, b]) = [ReLU(a), ReLU(b)]. C'est valide pour la plupart
                # des fonctions d'activation standard (ReLU, LeakyReLU, Sigmoid, Tanh...).
                activated_lower = self.activation_fn(x_lower)
                activated_upper = self.activation_fn(x_upper)
                
                x_interval = (activated_lower, activated_upper)

        y_lower_final, y_upper_final = x_interval
        
        # Squeeze si la sortie est scalaire
        if len(y_lower_final.shape) > 1 and y_lower_final.shape[1] == 1:
            y_lower_final = y_lower_final.squeeze(1)
            y_upper_final = y_upper_final.squeeze(1)
            
        return y_lower_final, y_upper_final
