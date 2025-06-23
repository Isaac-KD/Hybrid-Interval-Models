import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from typing_extensions import override

class Macsum():
    def __init__(self, N, phi_init=None):
        self.N = N
        self._phi = None
        self._update_phi(phi_init if phi_init is not None else np.random.randn(N))

    @property
    def phi(self):
        return self._phi
    
    def _update_phi(self, phi:np.ndarray):
        if phi.shape != (self.N,): raise ValueError(f"phi doit avoir la forme ({self.N},), reçu {phi.shape}")
        self._phi = phi
        self.phi_plus = np.maximum(phi, 0)
        self.phi_minus = np.minimum(phi, 0)
        self.perm_decreasing = np.argsort(self._phi)[::-1]
        self.perm_increasing = np.argsort(self._phi)
        
    def _partial_derivative_algo1(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Implémentation de l'Algorithme 1 de la page 19.
        Cette version suit la procédure décrite pas à pas.
        """
        n = self.N
        grad_y_lower = np.zeros(n, dtype=np.float64) # δy
        grad_y_upper = np.zeros(n, dtype=np.float64) # δȳ 

        perm_increasing = self.perm_increasing
        x_perm_inc = x[perm_increasing]

        # Calcul des min/max cumulés (xk et xk_bar)
        x_cumul_min = np.minimum.accumulate(x_perm_inc)
        x_cumul_max = np.maximum.accumulate(x_perm_inc)
        
        grad_y_lower[perm_increasing] = x_cumul_min
        grad_y_upper[perm_increasing] = x_cumul_max

        # δy_{ιk} = δy_{ιk} - x_{k-1} et δȳ_{ιk} = δȳ_{ιk} - x_bar_{k-1}
        grad_y_lower[perm_increasing[1:]] -= x_cumul_max[:-1]
        grad_y_upper[perm_increasing[1:]] -= x_cumul_min[:-1]
        
        perm_decreasing = self.perm_decreasing
        x_perm_dec = x[perm_decreasing]

        # Calcul des min/max cumulés
        x_cumul_min_dec = np.minimum.accumulate(x_perm_dec)
        x_cumul_max_dec = np.maximum.accumulate(x_perm_dec)
        
        # δy_{ιk} = δy_{ιk} + xk_bar et δȳ_{ιk} = δȳ_{ιk} + xk
        grad_y_lower[perm_decreasing] += x_cumul_min_dec 
        grad_y_upper[perm_decreasing] += x_cumul_max_dec 
        
        # δy_{ιk} = δy_{ιk} - x_bar_{k-1} et δȳ_{ιk} = δȳ_{ιk} - x_{k-1}
        grad_y_lower[perm_decreasing[1:]] -= x_cumul_min_dec[:-1]
        grad_y_upper[perm_decreasing[1:]] -= x_cumul_max_dec[:-1]

        # L'algorithme est terminé. grad_y_lower et grad_y_upper contiennent le résultat final.
        # L'ordre de retour dans le papier est (δy, δȳ), ce qui correspond à (y_lower, y_upper)
        return grad_y_upper, grad_y_lower
       
    def _partial_derivative(self,x):
        
        # implemention of the Proposition 5.3
        
        ranks_decreasing = np.empty(self.N, dtype=int)
        ranks_decreasing[self.perm_decreasing] = np.arange(self.N)
    
        ranks_increasing = np.empty(self.N, dtype=int)
        ranks_increasing[self.perm_increasing] = np.arange(self.N)
    
        x_b = x[self.perm_decreasing]
        x_d = x[self.perm_increasing]
        
        diff_max_xb = np.diff(np.maximum.accumulate(x_b), prepend=0)
        diff_min_xb = np.diff(np.minimum.accumulate(x_b), prepend=0)
        diff_max_xd = np.diff(np.maximum.accumulate(x_d), prepend=0)
        diff_min_xd = np.diff(np.minimum.accumulate(x_d), prepend=0)          
        
        return diff_max_xb[ranks_decreasing] + diff_min_xd[ranks_increasing],diff_min_xb[ranks_decreasing] + diff_max_xd[ranks_increasing]
    
    def _gradient_function_unit(self,x,y_true,prediction):
        
        # implementation of 5.2.2. Gradient descent p.15
        
        y_pred_upper,y_pred_lower = prediction
        d_upper, d_lower = self._partial_derivative(x)
        
        return 2*(y_pred_upper-y_true)*d_upper + 2*(y_pred_lower-y_true)*d_lower
    
    def _gradient_function_algo1(self,x,y_true,prediction):
        
        # implementation of 5.2.2. Gradient descent p.15
        
        y_pred_upper,y_pred_lower = prediction
        d_upper, d_lower = self._partial_derivative_algo1(x)
        
        return 2*(y_pred_upper-y_true)*d_upper + 2*(y_pred_lower-y_true)*d_lower
    

    def gradient(self, X, Y):
        """
        Calcule le gradient MOYEN de la perte sur l'ensemble du jeu de données (X, Y).
        """

        # implementation of 5.2.2. Gradient descent p.15
        
        all_prediction = [ self.prediction(x) for x in X]
        all_gradients = (self._gradient_function(X[i], Y[i], all_prediction[i]) for i in range(self.M))
                        
        loss = np.sum( self._loss(Y[i],all_prediction[i]) for i in range(self.M))
        return np.sum(list(all_gradients), axis=0)  / self.M, loss /self.M
    
    def prediction(self,x):
        """
        Calcule la borne supérieure et inférieur en implémentant la Proposition 5.1 (Équation 8 et 9).
        
        Args:
            x (np.ndarray): Le vecteur d'entrée de taille N.
            phi (np.ndarray): Le noyau de paramètres de taille N.

        Returns:
            float: La valeur de la borne supérieure _y et ỹ.
        """

        # Trier les vecteurs φ⁺, φ⁻ et x 
        
        phi_plus_sorted = self.phi_plus[self.perm_decreasing]
        phi_minus_sorted =self.phi_minus[self.perm_decreasing]
        x_permuted = x[self.perm_decreasing]

        # --- Calcul du premier grand terme de la somme (partie avec φ⁺) ---
        
        # La condition max(x[:0]) = 0 est gérée par `prepend=0`
        diff_running_max_x = np.diff(np.maximum.accumulate(x_permuted), prepend=0)
        diff_running_min_x = np.diff(np.minimum.accumulate(x_permuted), prepend=0)
        
        # Produit scalaire entre φ⁺ trié et les différences
        term_plus_for_upper = np.dot(phi_plus_sorted, diff_running_max_x)
        term_minus_for_upper = np.dot(phi_minus_sorted, diff_running_min_x)
        y_upper = term_plus_for_upper + term_minus_for_upper
        
        term_plus_for_lower = np.dot(phi_plus_sorted, diff_running_min_x)
        term_minus_for_lower = np.dot(phi_minus_sorted, diff_running_max_x)
        y_lower = term_plus_for_lower + term_minus_for_lower
        
        return y_lower,y_upper
    
    def _loss(self,y_true, pred) -> float:
        """Calcule la perte quadratique étendue pour un intervalle."""
        
        y_lower,y_upper = pred
        return (y_true - y_lower)**2 + (y_true - y_upper)**2
    
    def _loss_v2(self,y_true, pred) -> float:
        """Calcule la perte quadratique étendue pour un intervalle."""
        
        y_lower,y_upper = pred
        return alpha
    
    def loss(y_true, y_lower,y_upper) -> float:
        """Calcule la perte quadratique étendue pour un intervalle."""
        
        return (y_true - y_lower)**2 + (y_true - y_upper)**2

    def fit(self,X: np.ndarray,Y:np.array,X_eval: np.ndarray,Y_eval:np.array,alpha=1e-7,n_iteration=1000,epsilon=0.1):
        
        #================================================================================
        # Sécurité des entrée       
        #================================================================================
        if X.ndim != 2: raise ValueError(f"X doit avoir 2 dimensions (M, {self.N}), mais X.ndim = {X.ndim}")
        if X.shape[1] != self.N: raise ValueError(f"La deuxième dimension de X doit être {self.N} egale a la dimention de φ , mais X.shape[1] = {X.shape[1]}")
        if Y.ndim != 1: raise ValueError("Y doit être un vecteur (tableau 1D)")
        #================================================================================
        
        self.history = []
        self.M = X.shape[0]
        
        for i in trange(n_iteration,desc="Entraînement :"):
            grd, loss = self.gradient(X,Y)
            new_phi = self.phi - alpha*grd
            
            if np.linalg.norm(self.phi-new_phi) < epsilon:
                self._update_phi(new_phi)
                return self.phi
            
            self._update_phi(new_phi)
            if i%20 == 0: 
                #print(f"Iteration {i}, loss moyenne: {loss}, norme du gradient : {np.linalg.norm(grd)}")
                self.history.append(evaluate_model(X_eval, Y_eval, self))
        
        return self.phi
    
    def fit_adam(self, X: np.ndarray, Y: np.ndarray,
                 X_eval: np.ndarray, Y_eval: np.ndarray,
             learning_rate: float = 0.001, 
             n_epochs: int = 100, 
             batch_size: int = 32,
             beta1: float = 0.9, 
             beta2: float = 0.999, 
             epsilon_adam: float = 1e-8,
             epsilon_conv: float = 1e-6,
             print_every: int = 10):
        """
        Apprend le noyau φ en utilisant l'optimiseur Adam par mini-batch.

        Args:
            X (np.ndarray): Données d'entrée.
            Y (np.ndarray): Données cibles.
            learning_rate (float): Taux d'apprentissage pour Adam.
            n_epochs (int): Nombre de fois où l'on parcourt tout le jeu de données.
            batch_size (int): Taille des mini-batchs.
            beta1 (float): Paramètre de décroissance exponentielle pour le moment de premier ordre.
            beta2 (float): Paramètre de décroissance exponentielle pour le moment de second ordre.
            epsilon_adam (float): Petite constante pour éviter la division par zéro.
            epsilon_conv (float): Seuil de convergence sur la norme de la mise à jour.
            print_every (int): Fréquence d'affichage du statut (en époques).
        """
        # ================================================================================
        # Sécurité des entrées et initialisation
        # ================================================================================
        if X.ndim != 2: raise ValueError(f"X doit avoir 2 dimensions (M, {self.N}), mais X.ndim = {X.ndim}")
        if X.shape[1] != self.N: raise ValueError(f"La deuxième dimension de X doit être {self.N}, mais X.shape[1] = {X.shape[1]}")
        if Y.ndim != 1: raise ValueError("Y doit être un vecteur (tableau 1D)")
        
        n_samples = X.shape[0]
        self.history = []
        
        # Variables d'Adam
        m = np.zeros_like(self.phi, dtype=np.float64)  # Moment de premier ordre (moving average of gradients)
        v = np.zeros_like(self.phi, dtype=np.float64)  # Moment de second ordre (moving average of squared gradients)
        t = 0  # Compteur de pas de temps (itérations)

        print("Début de l'entraînement avec l'optimiseur Adam...")

        # Boucle sur les époques
        for epoch in trange(1, n_epochs + 1, desc="Entraînement Adam"):
            
            # Mélanger les données à chaque époque pour la stochasticité
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            
            epoch_losses = []

            # Boucle sur les mini-batchs
            for i in range(0, n_samples, batch_size):
                # Créer le mini-batch
                X_batch = X_shuffled[i : i + batch_size]
                Y_batch = Y_shuffled[i : i + batch_size]
                current_batch_size = len(X_batch)

                # --- Calcul du gradient sur le mini-batch ---
                grad_batch = np.zeros_like(self.phi, dtype=np.float64)
                loss_batch = 0.0
                for j in range(current_batch_size):
                    prediction_j = self.prediction(X_batch[j])
                    grad_batch += self._gradient_function_unit(X_batch[j], Y_batch[j], prediction_j)
                    loss_batch += self._loss(Y_batch[j], prediction_j)
                
                grad_moy_batch = grad_batch / current_batch_size
                epoch_losses.append(loss_batch / current_batch_size)
                
                t += 1 # Incrémenter le pas de temps

                # --- Mise à jour des paramètres avec Adam ---
                
                # 1. Mise à jour des moments
                m = beta1 * m + (1 - beta1) * grad_moy_batch
                v = beta2 * v + (1 - beta2) * (grad_moy_batch**2)
                
                # 2. Correction du biais des moments
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                
                # 3. Calcul de la mise à jour du noyau
                update_step = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon_adam)
                new_phi = self.phi - update_step

                # 4. Vérification de la convergence
                if np.linalg.norm(new_phi-self.phi) < epsilon_conv:
                    print(f"\nConvergence atteinte à l'époque {epoch}, itération {t}.")
                    self._update_phi(new_phi)
                    return self.phi

                # 5. Appliquer la mise à jour
                self._update_phi(new_phi)
                
            # Affichage du statut à la fin de certaines époques
            if epoch % print_every == 0:
                self.history.append(evaluate_model(X_eval, Y_eval, self))
                #print(f"Époque {epoch}/{n_epochs}, Loss moyenne: {avg_epoch_loss:.4f}")
                
        print("Entraînement terminé (nombre maximum d'époques atteint).")
        return self.phi
    
    def fit_adam_complet(self, X: np.ndarray, Y: np.ndarray,
                 X_eval: np.ndarray, Y_eval: np.ndarray,
             phi_true,
             learning_rate: float = 0.001, 
             n_epochs: int = 100,
             batch_size: int = 32,
             beta1: float = 0.9, 
             beta2: float = 0.999, 
             epsilon_adam: float = 1e-8,
             epsilon_conv: float = 1e-6,
             print_every: int = 10):
        """
        Apprend le noyau φ en utilisant l'optimiseur Adam par mini-batch.

        Args:
            X (np.ndarray): Données d'entrée.
            Y (np.ndarray): Données cibles.
            learning_rate (float): Taux d'apprentissage pour Adam.
            n_epochs (int): Nombre de fois où l'on parcourt tout le jeu de données.
            batch_size (int): Taille des mini-batchs.
            beta1 (float): Paramètre de décroissance exponentielle pour le moment de premier ordre.
            beta2 (float): Paramètre de décroissance exponentielle pour le moment de second ordre.
            epsilon_adam (float): Petite constante pour éviter la division par zéro.
            epsilon_conv (float): Seuil de convergence sur la norme de la mise à jour.
            print_every (int): Fréquence d'affichage du statut (en époques).
        """
        
        # ================================================================================
        # Sécurité des entrées et initialisation
        # ================================================================================
        
        if X.ndim != 2: raise ValueError(f"X doit avoir 2 dimensions (M, {self.N}), mais X.ndim = {X.ndim}")
        if X.shape[1] != self.N: raise ValueError(f"La deuxième dimension de X doit être {self.N}, mais X.shape[1] = {X.shape[1]}")
        if Y.ndim != 1: raise ValueError("Y doit être un vecteur (tableau 1D)")
        
        n_samples = X.shape[0]
        self.history = []
        
        # Variables d'Adam
        m = np.zeros_like(self.phi, dtype=np.float64)  # Moment de premier ordre (moving average of gradients)
        v = np.zeros_like(self.phi, dtype=np.float64)  # Moment de second ordre (moving average of squared gradients)
        t = 0  # Compteur de pas de temps (itérations)

        print("Début de l'entraînement avec l'optimiseur Adam...")

        # Boucle sur les époques
        for epoch in trange(1, n_epochs + 1, desc="Entraînement Adam"):
            
            # Mélanger les données à chaque époque pour la stochasticité
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            
            epoch_losses = []
            loss=0
            # Boucle sur les mini-batchs
            for i in range(0, n_samples, batch_size):
                # Créer le mini-batch
                X_batch = X_shuffled[i : i + batch_size]
                Y_batch = Y_shuffled[i : i + batch_size]
                current_batch_size = len(X_batch)

                # --- Calcul du gradient sur le mini-batch ---
                grad_batch = np.zeros_like(self.phi, dtype=np.float64)
                loss_batch = 0.0
                for j in range(current_batch_size):
                    prediction_j = self.prediction(X_batch[j])
                    grad_batch += self._gradient_function_unit(X_batch[j], Y_batch[j], prediction_j)
                    loss_batch += self._loss(Y_batch[j], prediction_j)
                
                grad_moy_batch = grad_batch / current_batch_size
                epoch_losses.append(loss_batch / current_batch_size)
                
                t += 1 # Incrémenter le pas de temps

                # --- Mise à jour des paramètres avec Adam ---
                
                # 1. Mise à jour des moments
                m = beta1 * m + (1 - beta1) * grad_moy_batch
                v = beta2 * v + (1 - beta2) * (grad_moy_batch**2)
                
                # 2. Correction du biais des moments
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                
                # 3. Calcul de la mise à jour du noyau
                update_step = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon_adam)
                new_phi = self.phi - update_step

                # 4. Vérification de la convergence
                if np.linalg.norm(new_phi-self.phi) < epsilon_conv:
                    print(f"\nConvergence atteinte à l'époque {epoch}, itération {t}.")
                    self._update_phi(new_phi)
                    return self.phi

                # 5. Appliquer la mise à jour
                self._update_phi(new_phi)
                
                loss+=loss_batch/current_batch_size
            # Affichage du statut à la fin de certaines époques
            if epoch % print_every == 0:
                val_metrics = evaluate_model_complet(X_eval, Y_eval, self,loss,phi_true)
                val_metrics['epoch'] = epoch
                self.history.append(val_metrics)
                
        print("Entraînement terminé (nombre maximum d'époques atteint).")
        return self.phi 
    
class Macsum_sigmoide(Macsum):
    
    def __init__(self, N,alpha=0.1,gamma=0.6,k_sigmoid=0.01, phi_init=None):
        super().__init__(N,phi_init)
        self.gamma = gamma
        self.alpha = alpha
        self.k_sigmoid=k_sigmoid
        
    @override
    def _gradient_function_algo1(self,x,y_true,prediction):
        
        # implementation of 5.2.2. Gradient descent p.15
        
        y_pred_upper,y_pred_lower = prediction
        d_upper, d_lower = self._partial_derivative_algo1(x)
        
        return self.alpha*( d_upper - d_lower) + self.gamma*(sigm_upper*(1-sigm_upper)*d_upper + sigm_lower*(1-sigm_lower)*d_lower)

    @override  
    def _gradient_function_unit(self, x, y_true, prediction):
        """
        Calcule le gradient de la fonction de coût combinée.
        Cette version est la dérivation correcte et complète, 
        en traitant correctement la règle de la chaîne pour les produits de fonctions.
        """
        y_pred_upper, y_pred_lower = prediction
        
        # d_upper = ∇y_upper 
        # d_lower = ∇y_lower
        d_upper, d_lower = self._partial_derivative(x)

        # --- Calcul des termes pour la pénalité ---
        diff_lower = y_pred_lower - y_true
        sig_l = sigmoide(diff_lower, self.k_sigmoid)
        sig_prime_l = sigmoide_prime(sig_l) * self.k_sigmoid
        
        diff_upper = y_true - y_pred_upper
        sig_u = sigmoide(diff_upper, self.k_sigmoid)
        sig_prime_u = sigmoide_prime(sig_u) * self.k_sigmoid

        # --- Assemblage des coefficients pour d_upper et d_lower ---
        
        # Coeff pour d_upper : vient de (α*y_upper)' et de (γ*sig(diff_u)*diff_u^2)'
        # La dérivée de la pénalité supérieure par rapport à y_upper est :
        # -γ * [diff_u^2 * sig'_u + 2*diff_u*sig_u]
        coeff_d_upper = self.alpha - self.gamma * ( (diff_upper**2) * sig_prime_u + 2 * diff_upper * sig_u )
        
        # Coeff pour d_lower : vient de (-α*y_lower)' et de (γ*sig(diff_l)*diff_l^2)'
        # La dérivée de la pénalité inférieure par rapport à y_lower est :
        # +γ * [diff_l^2 * sig'_l + 2*diff_l*sig_l]
        coeff_d_lower = -self.alpha + self.gamma * ( (diff_lower**2) * sig_prime_l + 2 * diff_lower * sig_l )

        # Gradient final
        final_gradient = (coeff_d_upper * d_upper) + (coeff_d_lower * d_lower)
        
        return final_gradient
    
    @override
    def gradient(self, X, Y):
        """
        Calcule le gradient MOYEN de la perte sur l'ensemble du jeu de données (X, Y).
        """

        # implementation of 5.2.2. Gradient descent p.15
        
        all_prediction = [ self.prediction(x) for x in X]
        all_gradients = (self._gradient_function_unit(X[i], Y[i], all_prediction[i]) for i in range(self.M))
                        
        loss = np.sum( self._loss(Y[i],all_prediction[i]) for i in range(self.M))
        return np.sum(list(all_gradients), axis=0)  / self.M, loss /self.M
    
    @override
    def _loss(self, y_true, pred) -> float:
        """ Perte = sig(diff) * diff^2 """
        y_lower, y_upper = pred
        
        diff_lower = y_lower - y_true
        diff_upper = y_true - y_upper
        
        loss_l = sigmoide(diff_lower, self.k_sigmoid) * (diff_lower**2)
        loss_u = sigmoide(diff_upper, self.k_sigmoid) * (diff_upper**2)
        
        return loss_l + loss_u

    
    @override
    def loss(y_true, y_lower,y_upper) -> float:
        """Calcule la perte quadratique étendue pour un intervalle."""
        
        return self._loss(y_true,(y_lower,y_upper))

def sigmoide(x,k=1):
    return  1 / (1 + np.exp(-1*(k * x)))

def sigmoide_prime(s_val):
    # La dérivée de sig(x) est sig(x) * (1 - sig(x))
    return s_val * (1 - s_val)
              
def evaluate_model(X: np.ndarray, Y: np.ndarray, macsum) -> dict:
    num_samples = X.shape[0]
    containment_count = 0
    total_distance = 0.0
    total_loss = 0.0
    total_spread = 0.0

    for i in range(num_samples):
        x = X[i]
        y_true = Y[i]
        
        y_lower, y_upper = macsum.prediction(x)
        
        if y_lower <= y_true <= y_upper:
            containment_count += 1
        
        total_distance += min(abs(y_true - y_lower), abs(y_true - y_upper))
        total_loss += Macsum.loss(y_true, y_lower, y_upper)
        total_spread += y_upper - y_lower
        
    return {
        "containment_rate": containment_count / num_samples,
        "avg_distance_to_interval": total_distance / num_samples,
        "avg_loss": total_loss / num_samples,
        "avg_interval_spread": total_spread / num_samples,
    }
    
def evaluate_model_complet(X: np.ndarray, Y: np.ndarray, macsum: Macsum,loss:float="no", phi_true: np.ndarray = None) -> dict:
    num_samples = X.shape[0]
    if num_samples == 0: return {}

    y_lower_preds, y_upper_preds = np.array([macsum.prediction(x) for x in X]).T
    
    if loss == "no":
        total_loss = np.mean([macsum._loss(Y[i], (y_lower_preds[i], y_upper_preds[i])) for i in range(num_samples)])
    else: total_loss = loss
    
    is_contained = (Y >= y_lower_preds) & (Y <= y_upper_preds)
    containment_rate = np.mean(is_contained)
    
    misclassified_indices = ~is_contained
    misclassified_count = np.sum(misclassified_indices)
    
    if misclassified_count > 0:
        dists = np.minimum(np.abs(Y[misclassified_indices] - y_lower_preds[misclassified_indices]),
                           np.abs(Y[misclassified_indices] - y_upper_preds[misclassified_indices]))
        avg_misp_distance = np.mean(dists)
    else:
        avg_misp_distance = 0.0
    
    spreads = y_upper_preds - y_lower_preds
    metrics = {
        "containment_rate": containment_rate,
        "avg_misp_distance": avg_misp_distance,
        "avg_loss": total_loss,
        "avg_interval_spread": np.mean(spreads),
        "median_interval_spread": np.median(spreads),
        "q95_interval_spread": np.quantile(spreads, 0.95)
    }

    if phi_true is not None:
        metrics["phi_distance_l2"] = np.linalg.norm(macsum.phi - phi_true)
        
    return metrics



def plot_metrics(history):
    """
    history : list of dicts, chaque dict contenant les clés :
        - 'containment_rate'
        - 'avg_distance_to_interval'
        - 'avg_loss'
        - 'avg_interval_spread'
    """
    iterations = [ i*10  for i in range(len(history))]  # Supposé que chaque dict correspond à un pas tous les 20 tours

    metrics = {
        "containment_rate": [],
        "avg_distance_to_interval": [],
        "avg_loss": [],
        "avg_interval_spread": []
    }

    # Extraire les valeurs
    for record in history:
        for key in metrics:
            metrics[key].append(record[key])

    # Plot
    plt.figure(figsize=(12, 8))
    for i, (metric_name, values) in enumerate(metrics.items(), 1):
        plt.subplot(3, 2, i)
        plt.plot(iterations, values, marker='o')
        plt.title(metric_name)
        plt.xlabel("Epoche")
        plt.grid(True)

    plt.tight_layout()
    plt.show()
    
def plot_metrics_complet(history_df): # Prend un DataFrame maintenant

    metrics_to_plot = [
        'avg_loss', 
        'containment_rate',
        'avg_interval_spread',
        'avg_misp_distance', 
        'phi_distance_l2',
        "median_interval_spread",
        "q95_interval_spread"  
    ]

    plt.figure(figsize=(12, 10))
    for i, metric_name in enumerate(metrics_to_plot, 1):
        if metric_name in history_df.columns:
            plt.subplot(4, 2, i)
            plt.plot(history_df['epoch'], history_df[metric_name], marker='.')
            plt.title(metric_name)
            plt.xlabel("Époque")
            plt.grid(True)
    
    plt.tight_layout()
    plt.show()