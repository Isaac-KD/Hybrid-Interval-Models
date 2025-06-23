import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from typing_extensions import override # For Python < 3.12, otherwise use typing.override
from torch.utils.data import DataLoader, TensorDataset # For batching in evaluation

# Determine device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class Macsum(nn.Module):
    def __init__(self, N, phi_init=None):
        super().__init__()
        self.N = N
        
        if phi_init is None:
            phi_init_tensor = torch.randn(N, device=DEVICE, dtype=torch.float64)
        elif isinstance(phi_init, np.ndarray):
            phi_init_tensor = torch.tensor(phi_init, device=DEVICE, dtype=torch.float64)
        else: 
            phi_init_tensor = phi_init.to(device=DEVICE, dtype=torch.float64)

        self._phi = nn.Parameter(phi_init_tensor) # _phi est le paramètre à apprendre
        
        # Ces attributs seront mis à jour par _recompute_phi_derived_attrs
        self.phi_plus = None
        self.phi_minus = None
        self.perm_decreasing = None
        self.perm_increasing = None
        self.ranks_decreasing_for_phi = None 
        self.ranks_increasing_for_phi = None 
        
        self._recompute_phi_derived_attrs()

    @property
    def phi(self):
        # Retourne une copie sur CPU pour éviter des modifications accidentelles
        return self._phi.data.clone().cpu().numpy()

    def _recompute_phi_derived_attrs(self):
        # Assure que _phi est bien 1D de taille N
        if self._phi.shape != (self.N,): 
            raise ValueError(f"Internal _phi must have shape ({self.N},), found {self._phi.shape}")
        
        # Utiliser .data ici est important pour l'approche du gradient manuel.
        # Les permutations sont considérées comme fixes pour un calcul de gradient donné.
        current_phi_values = self._phi.data 
        self.phi_plus = torch.clamp_min(current_phi_values, 0)
        self.phi_minus = torch.clamp_max(current_phi_values, 0)
        
        self.perm_decreasing = torch.argsort(current_phi_values, descending=True).long()
        self.perm_increasing = torch.argsort(current_phi_values, descending=False).long()

        self.ranks_decreasing_for_phi = torch.empty(self.N, dtype=torch.long, device=DEVICE)
        self.ranks_decreasing_for_phi[self.perm_decreasing] = torch.arange(self.N, device=DEVICE)

        self.ranks_increasing_for_phi = torch.empty(self.N, dtype=torch.long, device=DEVICE)
        self.ranks_increasing_for_phi[self.perm_increasing] = torch.arange(self.N, device=DEVICE)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_ndim = x.ndim
        if original_ndim == 1:
            x = x.unsqueeze(0) 
        
        batch_size, n_features = x.shape
        if n_features != self.N:
            raise ValueError(f"Input x feature dim must be {self.N}, got {n_features}")
        
        # Les attributs (phi_plus, perm_decreasing etc.) sont basés sur _phi.data
        # et sont donc traités comme des constantes du point de vue d'autograd pour _phi.
        x_permuted_by_phi_decreasing = x[:, self.perm_decreasing]
        
        phi_plus_sorted_decreasing = self.phi_plus[self.perm_decreasing]
        phi_minus_sorted_decreasing = self.phi_minus[self.perm_decreasing]

        acc_max_x_permuted = torch.cummax(x_permuted_by_phi_decreasing, dim=1).values
        acc_min_x_permuted = torch.cummin(x_permuted_by_phi_decreasing, dim=1).values
        
        padding = torch.zeros(batch_size, 1, dtype=x.dtype, device=x.device)

        padded_acc_max = torch.cat((padding, acc_max_x_permuted), dim=1)
        padded_acc_min = torch.cat((padding, acc_min_x_permuted), dim=1)

        diff_running_max_x = padded_acc_max[:, 1:] - padded_acc_max[:, :-1]
        diff_running_min_x = padded_acc_min[:, 1:] - padded_acc_min[:, :-1]
        
        term_plus_for_upper = torch.matmul(diff_running_max_x, phi_plus_sorted_decreasing)
        term_minus_for_upper = torch.matmul(diff_running_min_x, phi_minus_sorted_decreasing)
        y_upper = term_plus_for_upper + term_minus_for_upper 
        
        term_plus_for_lower = torch.matmul(diff_running_min_x, phi_plus_sorted_decreasing)
        term_minus_for_lower = torch.matmul(diff_running_max_x, phi_minus_sorted_decreasing)
        y_lower = term_plus_for_lower + term_minus_for_lower 
        
        if original_ndim == 1:
            return y_lower.squeeze(0), y_upper.squeeze(0)
        return y_lower, y_upper

    def _partial_derivative_torch_batch(self, X_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calcule les dérivées partielles de y_upper et y_lower par rapport à phi pour un batch.
        Basé sur la Proposition 5.3 du papier.
        X_batch: (batch_size, N)
        Returns: (d_y_upper_d_phi_batch, d_y_lower_d_phi_batch), both (batch_size, N)
        """
        batch_size = X_batch.shape[0]

        x_b = X_batch[:, self.perm_decreasing] 
        x_d = X_batch[:, self.perm_increasing]  

        padding = torch.zeros(batch_size, 1, dtype=X_batch.dtype, device=X_batch.device)

        acc_max_xb = torch.cummax(x_b, dim=1).values
        padded_acc_max_xb = torch.cat((padding, acc_max_xb), dim=1)
        diff_max_xb = padded_acc_max_xb[:, 1:] - padded_acc_max_xb[:, :-1] 

        acc_min_xb = torch.cummin(x_b, dim=1).values
        padded_acc_min_xb = torch.cat((padding, acc_min_xb), dim=1)
        diff_min_xb = padded_acc_min_xb[:, 1:] - padded_acc_min_xb[:, :-1] 

        acc_max_xd = torch.cummax(x_d, dim=1).values
        padded_acc_max_xd = torch.cat((padding, acc_max_xd), dim=1)
        diff_max_xd = padded_acc_max_xd[:, 1:] - padded_acc_max_xd[:, :-1] 

        acc_min_xd = torch.cummin(x_d, dim=1).values
        padded_acc_min_xd = torch.cat((padding, acc_min_xd), dim=1)
        diff_min_xd = padded_acc_min_xd[:, 1:] - padded_acc_min_xd[:, :-1] 
        
        d_upper_batch = diff_max_xb[:, self.ranks_decreasing_for_phi] + diff_min_xd[:, self.ranks_increasing_for_phi]
        d_lower_batch = diff_min_xb[:, self.ranks_decreasing_for_phi] + diff_max_xd[:, self.ranks_increasing_for_phi]
        
        return d_upper_batch, d_lower_batch


    def _get_phi_gradient_batch(self, X_batch: torch.Tensor, Y_batch: torch.Tensor,
                                y_lower_pred_batch: torch.Tensor, y_upper_pred_batch: torch.Tensor) -> torch.Tensor:
        """
        Calcule le gradient de la loss (par défaut MSE-like) par rapport à _phi pour un batch.
        Retourne le gradient moyen sur le batch.
        Cette méthode sera overridée par les sous-classes si elles ont une loss différente.
        """
        d_upper_batch, d_lower_batch = self._partial_derivative_torch_batch(X_batch) # (batch_size, N)

        # y_preds et Y_batch sont (batch_size,), il faut les unsqueeze pour le broadcast (batch_size, 1)
        term_upper = 2 * (y_upper_pred_batch - Y_batch).unsqueeze(1) * d_upper_batch # (batch_size, N)
        term_lower = 2 * (y_lower_pred_batch - Y_batch).unsqueeze(1) * d_lower_batch # (batch_size, N)
        
        phi_grads_batch = term_upper + term_lower # (batch_size, N)
        
        return phi_grads_batch.mean(dim=0) # Gradient moyen (N,)


    def _loss(self, y_true: torch.Tensor, pred: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # Calcule la loss par défaut (MSE-like) pour chaque échantillon du batch
        y_lower, y_upper = pred
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true, device=y_lower.device, dtype=y_lower.dtype)
        
        if y_true.ndim == 0 and y_lower.ndim == 1 and y_lower.shape[0] > 0 : # cas d'un scalaire y_true et batch de preds
             y_true = y_true.expand_as(y_lower)
        elif y_true.ndim == 1 and y_lower.ndim == 0 and y_true.shape[0] > 0: # cas batch y_true et scalaire preds
             y_lower = y_lower.expand_as(y_true)
             y_upper = y_upper.expand_as(y_true)
        
        loss_val = (y_true - y_lower).pow(2) + (y_true - y_upper).pow(2)
        return loss_val
    
    @staticmethod
    def loss_func(y_true: torch.Tensor, y_lower: torch.Tensor, y_upper: torch.Tensor) -> torch.Tensor:
        # Cette fonction statique est cohérente avec la loss de Macsum (non-sigmoide)
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true, device=y_lower.device, dtype=y_lower.dtype)
        return (y_true - y_lower).pow(2) + (y_true - y_upper).pow(2)

    def fit_adam(self, X_train: np.ndarray, Y_train: np.ndarray,
                 X_eval: np.ndarray, Y_eval: np.ndarray,
                 learning_rate: float = 0.001, 
                 n_epochs: int = 100, 
                 batch_size: int = 32,
                 beta1: float = 0.9, 
                 beta2: float = 0.999, 
                 epsilon_adam: float = 1e-8,
                 epsilon_conv: float = 1e-6,
                 print_every: int = 10,
                 phi_true_for_eval: np.ndarray = None):
        """ 
        Entraînement avec Adam et calcul MANUEL du gradient pour _phi.
        Cette méthode est héritée par les sous-classes.
        Elle utilise self._loss() pour calculer la perte (appel polymorphique).
        Elle utilise self._get_phi_gradient_batch() pour le gradient (appel polymorphique).
        """

        if X_train.ndim != 2 or X_train.shape[1] != self.N:
            raise ValueError(f"X_train shape error. Expected (M, {self.N}), got {X_train.shape}")
        if Y_train.ndim != 1:
            raise ValueError("Y_train must be 1D.")
        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError("X_train and Y_train must have the same number of samples.")

        X_train_t = torch.tensor(X_train, dtype=torch.float64, device=DEVICE)
        Y_train_t = torch.tensor(Y_train, dtype=torch.float64, device=DEVICE)
        
        train_dataset = TensorDataset(X_train_t, Y_train_t)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        self.to(DEVICE) 
        self.train() 

        optimizer = optim.Adam([self._phi], lr=learning_rate, betas=(beta1, beta2), eps=epsilon_adam)
        
        self.history = [] 

        print(f"Starting PyTorch Adam training ({type(self).__name__} with Manual Gradient for Phi)...")

        for epoch in trange(1, n_epochs + 1, desc=f"Adam Training ({type(self).__name__} Manual Grad)"):
            epoch_total_loss_sum = 0.0
            epoch_total_samples = 0

            for X_batch, Y_batch in train_dataloader:
                current_batch_size = X_batch.shape[0]
                if current_batch_size == 0: continue

                phi_before_step = self._phi.data.clone()
                optimizer.zero_grad()

                # 1. Forward pass
                y_lower_pred_batch, y_upper_pred_batch = self(X_batch)
                
                # 2. Calculer la loss (utilise la méthode _loss de l'INSTANCE, ex: MacsumSigmoidTorch._loss)
                batch_sample_losses = self._loss(Y_batch, (y_lower_pred_batch, y_upper_pred_batch))
                
                # 3. Calculer le gradient de _phi manuellement (utilise _get_phi_gradient_batch de l'INSTANCE)
                phi_gradient = self._get_phi_gradient_batch(X_batch, Y_batch, y_lower_pred_batch, y_upper_pred_batch)
                
                # 4. Assigner le gradient manuel à _phi.grad
                if self._phi.grad is None:
                    self._phi.grad = torch.zeros_like(self._phi.data)
                self._phi.grad.copy_(phi_gradient) 
                
                # 5. Mettre à jour les poids
                optimizer.step()          
                
                # 6. Recalculer les attributs dérivés de phi 
                self._recompute_phi_derived_attrs() 
                
                epoch_total_loss_sum += batch_sample_losses.sum().item()
                epoch_total_samples += current_batch_size

                phi_change_norm = torch.linalg.norm(self._phi.data - phi_before_step)
                if phi_change_norm < epsilon_conv and epoch > 1: # Avoid early stop at epoch 1
                    print(f"\nConvergence reached at epoch {epoch} (manual grad fit_adam for {type(self).__name__}). Change norm: {phi_change_norm:.2e}")
                    if print_every > 0 and X_eval is not None and Y_eval is not None:
                         avg_epoch_loss_train = epoch_total_loss_sum / epoch_total_samples if epoch_total_samples > 0 else float('nan')
                         val_metrics = evaluate_model_complet(X_eval, Y_eval, self, 
                                                              avg_epoch_loss_train, 
                                                              phi_true_for_eval, batch_size=batch_size)
                         val_metrics['epoch'] = epoch
                         self.history.append(val_metrics)
                    return self.phi 
            
            avg_epoch_loss_train = epoch_total_loss_sum / epoch_total_samples if epoch_total_samples > 0 else float('nan')
            if print_every > 0 and epoch % print_every == 0:
                if X_eval is not None and Y_eval is not None:
                    val_metrics = evaluate_model_complet(X_eval, Y_eval, self, avg_epoch_loss_train, phi_true_for_eval, batch_size=batch_size)
                    val_metrics['epoch'] = epoch
                    self.history.append(val_metrics)
        
        print(f"Training finished for {type(self).__name__} (max epochs reached or converged earlier via manual grad fit_adam).")
        return self.phi

# --- Fonctions utilitaires pour MacsumSigmoidTorch ---
def sigmoide_torch(x: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    return torch.sigmoid(k * x) # PyTorch a une fonction sigmoide optimisée


class MacsumSigmoidTorch(Macsum):
    def __init__(self, N, alpha: float = 0.2, gamma: float = 1.0, k_sigmoid: float = 0.5, phi_init=None):
        super().__init__(N, phi_init)
        self.alpha = alpha       # Poids pour la largeur de l'intervalle
        self.gamma = gamma       # Poids pour la pénalité de non-contenu
        self.k_sigmoid = k_sigmoid # Raideur de la sigmoïde pour la pénalité
        
    @override
    def _loss(self, y_true: torch.Tensor, pred: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """ 
        Calcule la fonction de coût combinée pour MacsumSigmoidTorch pour chaque échantillon.
        Loss = α * (y_upper - y_lower) 
               + γ * [ sig(k*(y_lower - y_true)) * (y_lower - y_true)^2 
                     + sig(k*(y_true - y_upper)) * (y_true - y_upper)^2 ]
        """
        y_lower_pred, y_upper_pred = pred

        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true, device=y_lower_pred.device, dtype=y_lower_pred.dtype)
        
        if y_true.ndim == 0 and y_lower_pred.ndim == 1 and y_lower_pred.shape[0] > 0 : 
             y_true = y_true.expand_as(y_lower_pred)
        elif y_true.ndim == 1 and y_lower_pred.ndim == 0 and y_true.shape[0] > 0: 
             y_lower_pred = y_lower_pred.expand_as(y_true)
             y_upper_pred = y_upper_pred.expand_as(y_true)

        loss_spread = self.alpha * (y_upper_pred - y_lower_pred)

        diff_lower = y_lower_pred - y_true  # Positif si y_lower_pred > y_true (mauvais)
        diff_upper = y_true - y_upper_pred  # Positif si y_true > y_upper_pred (mauvais)
        
        penalty_lower = sigmoide_torch(diff_lower, self.k_sigmoid) * (diff_lower.pow(2))
        penalty_upper = sigmoide_torch(diff_upper, self.k_sigmoid) * (diff_upper.pow(2))
        
        loss_penalty = self.gamma * (penalty_lower + penalty_upper)
        
        combined_loss_per_sample = loss_spread + loss_penalty
        return combined_loss_per_sample

    @staticmethod
    def loss_func_sigmoid(y_true: torch.Tensor, y_lower: torch.Tensor, y_upper: torch.Tensor,
                          alpha: float, gamma: float, k_sigmoid: float) -> torch.Tensor:
        """ Fonction de coût statique pour MacsumSigmoidTorch, utilisable en externe. """
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true, device=y_lower.device, dtype=y_lower.dtype)

        loss_spread = alpha * (y_upper - y_lower)

        diff_lower = y_lower - y_true
        diff_upper = y_true - y_upper
        
        penalty_lower = sigmoide_torch(diff_lower, k_sigmoid) * (diff_lower.pow(2))
        penalty_upper = sigmoide_torch(diff_upper, k_sigmoid) * (diff_upper.pow(2))
        
        loss_penalty = gamma * (penalty_lower + penalty_upper)
        
        return loss_spread + loss_penalty

    @override
    def _get_phi_gradient_batch(self, X_batch: torch.Tensor, Y_batch: torch.Tensor,
                                y_lower_pred_batch: torch.Tensor, y_upper_pred_batch: torch.Tensor) -> torch.Tensor:
        """
        Calcule le gradient de la loss sigmoïde (définie dans self._loss de cette classe) 
        par rapport à _phi pour un batch. Retourne le gradient moyen sur le batch.
        dL/dphi = (dL/dy_upper * dy_upper/dphi) + (dL/dy_lower * dy_lower/dphi)
        """
        # dy_upper/dphi et dy_lower/dphi sont donnés par _partial_derivative_torch_batch
        # Shapes: (batch_size, N)
        grad_phi_y_upper_batch, grad_phi_y_lower_batch = self._partial_derivative_torch_batch(X_batch)

        # Y_batch, y_lower_pred_batch, y_upper_pred_batch shapes: (batch_size,)
        
        # Calcul de z_l = y_lower_pred - y_true et z_u = y_true - y_upper_pred
        z_l = y_lower_pred_batch - Y_batch
        z_u = Y_batch - y_upper_pred_batch

        # s_l = sig(k*z_l), s_u = sig(k*z_u)
        s_l = sigmoide_torch(z_l, self.k_sigmoid)
        s_u = sigmoide_torch(z_u, self.k_sigmoid)

        # Calcul de dL/dy_lower_pred_batch et dL/dy_upper_pred_batch
        # L = alpha * (y_upper - y_lower) + gamma * (P_l + P_u)
        # P_l = sig(k*z_l) * z_l^2  => dP_l/dz_l = k*s_l*(1-s_l)*z_l^2 + 2*z_l*s_l
        # P_u = sig(k*z_u) * z_u^2  => dP_u/dz_u = k*s_u*(1-s_u)*z_u^2 + 2*z_u*s_u
        
        # dL/dy_lower = -alpha + gamma * (dP_l/dz_l) * (dz_l/dy_lower)
        # dz_l/dy_lower = 1
        term_penalty_deriv_lower = self.gamma * (
            self.k_sigmoid * s_l * (1 - s_l) * z_l.pow(2) + 2 * z_l * s_l
        )
        dL_d_y_lower_batch = -self.alpha + term_penalty_deriv_lower
        
        # dL/dy_upper = alpha + gamma * (dP_u/dz_u) * (dz_u/dy_upper)
        # dz_u/dy_upper = -1
        term_penalty_deriv_upper = -self.gamma * (
            self.k_sigmoid * s_u * (1 - s_u) * z_u.pow(2) + 2 * z_u * s_u
        )
        dL_d_y_upper_batch = self.alpha + term_penalty_deriv_upper

        # Combinaison pour le gradient de phi
        # Unsqueeze pour broadcast: (batch_size, 1) * (batch_size, N) -> (batch_size, N)
        phi_grads_for_batch = (
            dL_d_y_upper_batch.unsqueeze(1) * grad_phi_y_upper_batch +
            dL_d_y_lower_batch.unsqueeze(1) * grad_phi_y_lower_batch
        ) 
        
        return phi_grads_for_batch.mean(dim=0) # Gradient moyen (N,)

    # Pas besoin d'overrider fit_adam, elle est héritée de Macsum et fonctionnera correctement
    # grâce au polymorphisme de _loss et _get_phi_gradient_batch.

def evaluate_model_complet(X_np: np.ndarray, Y_np: np.ndarray, 
                           macsum_model: Macsum, # Peut être Macsum ou MacsumSigmoidTorch
                           current_train_loss: float = float('nan'), 
                           phi_true_np: np.ndarray = None,
                           batch_size: int = 32) -> dict:
    macsum_model.eval() 

    X_t = torch.tensor(X_np, dtype=torch.float64, device=DEVICE)
    Y_t = torch.tensor(Y_np, dtype=torch.float64, device=DEVICE)
    
    eval_dataset = TensorDataset(X_t, Y_t)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    num_samples = len(eval_dataset)
    if num_samples == 0: return {
        "containment_rate": 0.0, "avg_misp_distance": 0.0, "avg_eval_loss": float('nan'),
        "avg_train_loss": current_train_loss, "avg_interval_spread": 0.0,
        "median_interval_spread": 0.0, "q95_interval_spread": 0.0,
        "phi_distance_l2": float('nan') if phi_true_np is not None else None
    }

    all_y_lower_preds = []
    all_y_upper_preds = []
    total_eval_loss_sum = 0.0

    with torch.no_grad():
        for X_batch, Y_batch in eval_dataloader:
            y_lower_batch, y_upper_batch = macsum_model(X_batch) 
            
            all_y_lower_preds.append(y_lower_batch.cpu())
            all_y_upper_preds.append(y_upper_batch.cpu())
            
            # Utilise la méthode _loss de l'instance macsum_model (polymorphique)
            batch_eval_losses = macsum_model._loss(Y_batch, (y_lower_batch, y_upper_batch))
            total_eval_loss_sum += batch_eval_losses.sum().item()

    y_lower_preds_np = torch.cat(all_y_lower_preds).numpy()
    y_upper_preds_np = torch.cat(all_y_upper_preds).numpy()

    avg_loss_eval = total_eval_loss_sum / num_samples if num_samples > 0 else float('nan')
    
    is_contained = (Y_np >= y_lower_preds_np) & (Y_np <= y_upper_preds_np)
    containment_rate = np.mean(is_contained) if num_samples > 0 else 0.0
    
    misclassified_indices = ~is_contained
    misclassified_count = np.sum(misclassified_indices)
    
    avg_misp_distance = 0.0
    if misclassified_count > 0:
        dists = np.minimum(np.abs(Y_np[misclassified_indices] - y_lower_preds_np[misclassified_indices]),
                           np.abs(Y_np[misclassified_indices] - y_upper_preds_np[misclassified_indices]))
        avg_misp_distance = np.mean(dists)
    
    spreads = y_upper_preds_np - y_lower_preds_np
    metrics = {
        "containment_rate": containment_rate,
        "avg_misp_distance": avg_misp_distance,
        "avg_eval_loss": avg_loss_eval, 
        "avg_train_loss": current_train_loss, 
        "avg_interval_spread": np.mean(spreads) if num_samples > 0 else 0.0,
        "median_interval_spread": np.median(spreads) if num_samples > 0 else 0.0,
        "q95_interval_spread": np.quantile(spreads, 0.95) if num_samples > 0 and len(spreads) > 0 else 0.0
    }

    if phi_true_np is not None:
        phi_model_np = macsum_model.phi 
        metrics["phi_distance_l2"] = np.linalg.norm(phi_model_np - phi_true_np)
    else:
        metrics["phi_distance_l2"] = None

    return metrics

def plot_metrics_complet(history_list_of_dicts):
    if not history_list_of_dicts:
        print("No history to plot.")
        return

    all_keys = set()
    for d in history_list_of_dicts:
        all_keys.update(d.keys())
    
    history_dict = {key: [dic.get(key) for dic in history_list_of_dicts] 
                    for key in all_keys}
    
    epochs = history_dict.get('epoch', list(range(len(history_list_of_dicts))))
    if not epochs or not any(e is not None for e in epochs) : 
        epochs = list(range(len(history_list_of_dicts)))


    metrics_to_plot = [
        'avg_train_loss', 'avg_eval_loss', 'containment_rate',
        'avg_interval_spread', 'avg_misp_distance', 'phi_distance_l2',
        "median_interval_spread", "q95_interval_spread"  
    ]

    plottable_metrics_data = {}
    for metric_name in metrics_to_plot:
        if metric_name in history_dict:
            values = history_dict[metric_name]
            valid_indices = [i for i, val in enumerate(values) 
                             if val is not None and not (isinstance(val, float) and np.isnan(val))]
            
            if valid_indices:
                current_epochs = [epochs[i] for i in valid_indices if i < len(epochs)] # ensure epoch index is valid
                current_values = [values[i] for i in valid_indices]
                valid_epochs_values = [(ep, val) for ep, val in zip(current_epochs, current_values) if ep is not None]
                if valid_epochs_values:
                    plottable_metrics_data[metric_name] = (
                        [item[0] for item in valid_epochs_values], 
                        [item[1] for item in valid_epochs_values]
                    )

    num_plots = len(plottable_metrics_data)
    if num_plots == 0:
        print("No plottable metrics with valid data found in history.")
        return
        
    cols = 2
    rows = (num_plots + cols - 1) // cols

    plt.figure(figsize=(6 * cols, 4 * rows))
    plot_idx = 1
    for metric_name, (valid_epochs, valid_values) in plottable_metrics_data.items():
         plt.subplot(rows, cols, plot_idx)
         plt.plot(valid_epochs, valid_values, marker='.')
         plt.title(metric_name.replace("_", " ").title())
         plt.xlabel("Époque")
         plt.grid(True)
         plot_idx += 1
    
    plt.tight_layout()
    plt.show()