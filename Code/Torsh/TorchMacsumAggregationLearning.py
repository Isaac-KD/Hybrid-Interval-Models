import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

from tqdm import trange
from typing_extensions import override # For Python < 3.12
from typing import Type,List, Dict, Any, Optional
from torch.utils.data import DataLoader, TensorDataset # For batching in evaluation
from sklearn.model_selection import KFold

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

        # _phi est le paramètre à apprendre, il a requires_grad=True par défaut
        self._phi = nn.Parameter(phi_init_tensor) 
        
        # NOTE : On ne pré-calcule plus les attributs ici, car ils doivent être
        # calculés à la volée dans forward() pour être dans le graphe de calcul.

    @property
    def phi(self):
        return self._phi.data.clone().cpu().numpy()

    def _recompute_phi_derived_attrs(self):
        pass # Ne fait plus rien, tout est dans forward.


    @override
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # --- DÉBUT DE LA CORRECTION MAJEURE ---
        original_ndim = x.ndim
        if original_ndim == 1:
            x = x.unsqueeze(0) 
        
        batch_size, n_features = x.shape
        if n_features != self.N:
            raise ValueError(f"Input x feature dim must be {self.N}, got {n_features}")
        
        # --- CALCULS FAITS À LA VOLÉE DANS FORWARD ---
        # 1. Calculer les permutations (traitées comme des constantes par autograd)
        # On utilise un bloc no_grad pour être explicite que le tri ne doit pas avoir de gradient.
        with torch.no_grad():
            perm_decreasing = torch.argsort(self._phi, descending=True).long()

        # 2. Calculer phi_plus et phi_minus EN GARDANT LA CONNEXION AU GRAPHE
        # On n'utilise PAS .data ou .detach() ici !
        phi_plus = torch.clamp_min(self._phi, 0)
        phi_minus = torch.clamp_max(self._phi, 0)

        # 3. Appliquer la permutation à x et aux versions de phi
        x_permuted_by_phi_decreasing = x[:, perm_decreasing]
        phi_plus_sorted_decreasing = phi_plus[perm_decreasing]
        phi_minus_sorted_decreasing = phi_minus[perm_decreasing]

        acc_max_x_permuted = torch.cummax(x_permuted_by_phi_decreasing, dim=1).values
        acc_min_x_permuted = torch.cummin(x_permuted_by_phi_decreasing, dim=1).values
        
        padding = torch.zeros(batch_size, 1, dtype=x.dtype, device=x.device)

        padded_acc_max = torch.cat((padding, acc_max_x_permuted), dim=1)
        padded_acc_min = torch.cat((padding, acc_min_x_permuted), dim=1)

        diff_running_max_x = padded_acc_max[:, 1:] - padded_acc_max[:, :-1]
        diff_running_min_x = padded_acc_min[:, 1:] - padded_acc_min[:, :-1]
        
        # Ces matmul vont maintenant propager le gradient vers phi_plus et phi_minus,
        # et donc vers self._phi.
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
        
        with torch.no_grad():
            perm_decreasing = torch.argsort(self._phi, descending=True).long()
            perm_increasing = torch.argsort(self._phi, descending=False).long()

            ranks_decreasing_for_phi = torch.empty(self.N, dtype=torch.long, device=DEVICE)
            ranks_decreasing_for_phi[perm_decreasing] = torch.arange(self.N, device=DEVICE)

            ranks_increasing_for_phi = torch.empty(self.N, dtype=torch.long, device=DEVICE)
            ranks_increasing_for_phi[perm_increasing] = torch.arange(self.N, device=DEVICE)
        
        batch_size = X_batch.shape[0]

        x_b = X_batch[:, perm_decreasing] 
        x_d = X_batch[:, perm_increasing]  

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
        
        d_upper_batch = diff_max_xb[:, ranks_decreasing_for_phi] + diff_min_xd[:, ranks_increasing_for_phi]
        d_lower_batch = diff_min_xb[:, ranks_decreasing_for_phi] + diff_max_xd[:, ranks_increasing_for_phi]
        
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
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true, device=y_lower.device, dtype=y_lower.dtype)
        return (y_true - y_lower).pow(2) + (y_true - y_upper).pow(2)

    def fit_adam(self, X_train: np.ndarray, Y_train: np.ndarray,
                 X_eval: np.ndarray = None, Y_eval: np.ndarray=None,
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


class MacsumSigmoidTorchV2(Macsum):
    def __init__(self, N, alpha: float = 0.2,  gamma_lower: float = 1.0,gamma_upper:float = 1.0, k_sigmoid: float = 0.5, phi_init=None):
        super().__init__(N, phi_init)
        self.alpha = alpha       # Poids pour la largeur de l'intervalle
        self.gamma_lower = gamma_lower      # Poids pour la pénalité de non-contenu
        self.gamma_upper = gamma_upper
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
        
        loss_penalty = self.gamma_lower * penalty_lower +  self.gamma_upper * penalty_upper
        
        combined_loss_per_sample = loss_spread + loss_penalty
        return combined_loss_per_sample

    @staticmethod
    def loss_func_sigmoid(y_true: torch.Tensor, y_lower: torch.Tensor, y_upper: torch.Tensor,
                          alpha: float, gamma: float, k_sigmoid: float) -> torch.Tensor:
        """ Fonction de coût statique pour MacsumSigmoidTorch, utilisable en externe. """
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true, device=y_lower.device, dtype=y_lower.dtype)

        loss_spread = alpha * (y_upper - y_lower)**2

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
        term_penalty_deriv_lower = self.gamma_lower * (
            self.k_sigmoid * s_l * (1 - s_l) * z_l.pow(2) + 2 * z_l * s_l
        )
        dL_d_y_lower_batch = -self.alpha + term_penalty_deriv_lower
        
        # dL/dy_upper = alpha + gamma * (dP_u/dz_u) * (dz_u/dy_upper)
        # dz_u/dy_upper = -1
        term_penalty_deriv_upper = -self.gamma_upper * (
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
    
    
def cross_validate_macsum(
    model_class: Type[Macsum],
    X: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 5,
    n_epochs:int = 100,
    batch_size:int = 64,
    learning_rate:float = 1e-3,
    epsilon_conv:float = 1e-4,
    phi_true:np.ndarray = None,
    beta1:float = 0.8,
    beta2:float=0.999,
    random_state: int = 42,
) -> tuple[dict, list[dict]]:
    """
    Effectue une validation croisée k-fold pour un modèle de type Macsum.

    Args:
        model_class (Type[Macsum]): La classe du modèle à utiliser (ex: Macsum ou MacsumSigmoidTorch).
        model_params (dict): Dictionnaire des paramètres pour l'initialisation du modèle 
                             (ex: {'N': ..., 'alpha': ...}).
        fit_params (dict): Dictionnaire des paramètres pour la méthode `fit_adam`
                           (ex: {'learning_rate': ..., 'n_epochs': ...}).
        X_data (np.ndarray): L'ensemble des caractéristiques (features).
        Y_data (np.ndarray): L'ensemble des cibles (labels).
        n_splits (int): Le nombre de plis (k) pour la validation croisée.
        random_state (int): Graine aléatoire pour la reproductibilité du découpage.
        verbose (bool): Si True, affiche la progression et les résultats de chaque pli.

    Returns:
        tuple[dict, list[dict]]: 
        - Un dictionnaire contenant les métriques moyennes et l'écart-type sur tous les plis.
        - Une liste de dictionnaires, où chaque dictionnaire contient les métriques pour un pli.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []
    for fold_idx, (train_index, val_index) in enumerate(kf.split(X)):
        print(" tour :",fold_idx," / ", n_splits)
        model = copy.deepcopy(model_class)
        model.fit_adam(X[train_index],Y[train_index],X[ val_index],Y[ val_index],n_epochs=n_epochs,batch_size=batch_size,phi_true_for_eval=phi_true,learning_rate=learning_rate,epsilon_conv=epsilon_conv,beta1=beta1,beta2=beta2)
        fold_results.append(model.history)
    return fold_results
    

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
        metrics["phi_distance_l2_normal"] = np.linalg.norm(phi_model_np - phi_true_np)/macsum_model.N
    else:
        metrics["phi_distance_l2_normal"] = None

    return metrics

def plot_metrics_complet(list_of_histories: List[List[Dict[str, Any]]],title: str = "Résultats agrégés de la validation croisée" ):
    """
    Trace les métriques agrégées (moyenne et écart-type) à partir d'une liste
    d'historiques provenant d'une validation croisée.
    """
    if not list_of_histories or not any(h for h in list_of_histories):
        print("La liste d'historiques est vide ou ne contient que des listes vides.")
        return

    all_data = []
    for fold_idx, history in enumerate(list_of_histories):
        if not history: continue
        for point in history:
            record = point.copy()
            record['fold'] = fold_idx
            all_data.append(record)
    
    if not all_data:
        print("Aucun point de données valide trouvé dans les historiques.")
        return
        
    df = pd.DataFrame(all_data)

    metrics_to_plot = [
        'avg_train_loss', 'avg_eval_loss', 'containment_rate',
        'avg_interval_spread', 'avg_misp_distance', 'phi_distance_l2_normal',
        'median_interval_spread', 'q95_interval_spread'
    ]
    
    available_metrics = [m for m in metrics_to_plot if m in df.columns]

    if 'epoch' not in df.columns:
        print("Erreur : La clé 'epoch' est manquante.")
        return

    for metric in available_metrics:
        df[metric] = pd.to_numeric(df[metric], errors='coerce')

    aggregated_stats = df.groupby('epoch')[available_metrics].agg(['mean', 'std'])

    num_plots = len(available_metrics)
    if num_plots == 0:
        print("Aucune des métriques à tracer n'a été trouvée.")
        return
        
    cols = 2
    rows = (num_plots + cols - 1) // cols
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for i, metric_name in enumerate(available_metrics):
        ax = axes[i]
        
        # Le .dropna() est une sécurité supplémentaire pour éviter de tracer des points
        stats = aggregated_stats[metric_name].dropna()
        if stats.empty:
            ax.set_title(f"{metric_name.replace('_', ' ').title()}\n(Pas de données valides)", fontsize=12)
            continue
            
        epochs = stats.index
        mean_values = stats['mean']
        std_values = stats['std'].fillna(0) # std peut être NaN si un seul pli a des données pour une époque

        ax.plot(epochs, mean_values, marker='o', markersize=4, linestyle='-', label='Moyenne sur les plis')
        ax.fill_between(
            epochs,
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.2,
            label='± 1 écart-type'
        )
        
        ax.set_title(metric_name.replace("_", " ").title(), fontsize=12)
        ax.set_xlabel("Époque")
        ax.set_ylabel("Valeur")
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    for i in range(num_plots, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
def plot_multiple_histories_by_key ( list_of_histories: List[List[Dict[str, Any]]],
    key_to_plot: str,
    aggregate: bool = False,
    labels: Optional[List[str]] = None,
    reference_value: Optional[float] = None,
    title: Optional[str] = None,
    xlabel: str = "Époque",
    ylabel: str = "Valeur"
):
    """
    Trace l'évolution d'une métrique pour plusieurs historiques.
    Peut soit tracer chaque historique séparément, soit tracer leur moyenne agrégée.

    Args:
        list_of_histories (List[List[Dict]]): 
            Liste d'historiques (ex: sortie de validation croisée).
        key_to_plot (str): 
            La clé de la métrique à tracer.
        aggregate (bool): 
            Si True, trace la moyenne et l'écart-type de tous les historiques.
            Si False, trace une ligne distincte pour chaque historique.
        labels (Optional[List[str]]): 
            Étiquettes pour chaque courbe (utilisé uniquement si aggregate=False).
        reference_value (Optional[float]): 
            Trace une ligne de référence horizontale.
        title (Optional[str]): 
            Titre du graphique.
        xlabel (str), ylabel (str): Étiquettes des axes.
    """
    # --- 1. Préparation robuste des données avec Pandas ---
    if not list_of_histories or not any(h for h in list_of_histories):
        print("La liste d'historiques est vide. Impossible de tracer.")
        return

    all_data = []
    for fold_idx, history in enumerate(list_of_histories):
        if not history: continue
        for point in history:
            record = point.copy()
            record['fold'] = fold_idx
            all_data.append(record)
    
    if not all_data:
        print(f"Aucune donnée trouvée pour la clé '{key_to_plot}'.")
        return
        
    df = pd.DataFrame(all_data)

    # Vérification et nettoyage des colonnes nécessaires
    if key_to_plot not in df.columns or 'epoch' not in df.columns:
        print(f"Les clés 'epoch' ou '{key_to_plot}' sont manquantes. Impossible de tracer.")
        return
    df[key_to_plot] = pd.to_numeric(df[key_to_plot], errors='coerce')
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    df.dropna(subset=[key_to_plot, 'epoch'], inplace=True)
    if df.empty:
        print(f"Aucune donnée valide trouvée pour la clé '{key_to_plot}' après nettoyage.")
        return

    # --- 2. Configuration du graphique ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # --- 3. Logique de traçage (agrégé ou séparé) ---
    if aggregate:
        # --- CAS AGRÉGÉ (le style que vous aimez) ---
        stats = df.groupby('epoch')[key_to_plot].agg(['mean', 'std'])
        epochs = stats.index
        mean_values = stats['mean']
        std_values = stats['std'].fillna(0)

        ax.plot(epochs, mean_values, marker='o', markersize=4, linestyle='-', label=f"Mean of '{key_to_plot}'")
        ax.fill_between(
            epochs, mean_values - std_values, mean_values + std_values,
            alpha=0.2, label='± 1 standard deviation across cross-validation folds'
        )
    else:
        # --- CAS LIGNES SÉPARÉES (comportement original amélioré) ---
        if labels is None:
            labels = [f"Courbe {i+1}" for i in range(len(list_of_histories))]
        
        for fold_id in sorted(df['fold'].unique()):
            fold_df = df[df['fold'] == fold_id]
            label = labels[int(fold_id)] if int(fold_id) < len(labels) else f"Courbe {int(fold_id)+1}"
            ax.plot(fold_df['epoch'], fold_df[key_to_plot], marker='.', linestyle='-', label=label)

    # --- 4. Finalisation du graphique ---
    if reference_value is not None:
        ax.axhline(y=reference_value, color='crimson', linestyle='--', label=f"Référence with True Kernel = {reference_value:.2f}")

    if title is None:
        agg_str = "Agrégée" if aggregate else "Individuelle"
        title = f"Évolution {agg_str} de {key_to_plot.replace('_', ' ').title()}"
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()