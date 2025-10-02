"""
Modèles de réseaux de neurones pour l'agent SAC simplifié.
Implémente Actor et Critic sans mécanisme d'attention pour performances optimales.
Version compatible avec modelisation.pdf Section 2.5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any
import math

from config import Config


class SimpleActor(nn.Module):
    """
    Réseau Actor simplifié pour SAC selon modelisation.pdf Section 2.5.
    Architecture efficace sans attention pour performances optimales.
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Réseau principal
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Têtes pour moyenne et log-variance
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # Contraintes pour log_std
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass retournant moyenne et log-std pour chaque action
        """
        features = self.network(state)
        
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Contraindre log_std dans une plage raisonnable
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Échantillonnage avec reparameterization trick (Équation 18)
        """
        mean, log_std = self.forward(state)
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            std = log_std.exp()
            
            # Reparameterization trick
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # rsample pour gradient
            
            # Transformation tanh (Équation 18)
            action = torch.tanh(x_t)
            
            # Calcul log-probabilité avec correction Jacobienne
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob


class SimpleCritic(nn.Module):
    """
    Réseau Critic simplifié pour SAC selon modelisation.pdf Section 2.5.
    Implémente Q(s,a) avec architecture efficace.
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dim: int = 256):
        super().__init__()
        
        # Réseau Q-value
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass Q(s,a)
        """
        x = torch.cat([state, action], dim=1)
        return self.network(x)


# Alias pour compatibilité avec le code existant
class ActorWithAttention(SimpleActor):
    """
    Wrapper de compatibilité - utilise maintenant SimpleActor sans attention
    Remplace l'ancienne implémentation complexe avec attention
    """
    
    def __init__(self, 
                 num_assets: int,
                 feature_dim: int = Config.FEATURE_DIM,
                 hidden_dim: int = Config.HIDDEN_DIM,
                 attention_heads: int = Config.ATTENTION_HEADS):
        
        # Calculer la dimension d'état selon l'espace d'état amélioré
        state_dim = 339  # Dimension de l'espace d'état amélioré
        action_dim = num_assets
        
        super().__init__(state_dim, action_dim, hidden_dim)
        
        print(f"📊 SAC Actor simplifié initialisé: {state_dim}→{action_dim} (sans attention)")


class CriticWithAttention(SimpleCritic):
    """
    Wrapper de compatibilité - utilise maintenant SimpleCritic sans attention
    Remplace l'ancienne implémentation complexe avec attention
    """
    
    def __init__(self, 
                 num_assets: int,
                 feature_dim: int = Config.FEATURE_DIM,
                 hidden_dim: int = Config.HIDDEN_DIM,
                 attention_heads: int = Config.ATTENTION_HEADS):
        
        # Calculer les dimensions
        state_dim = 339  # Dimension de l'espace d'état amélioré
        action_dim = num_assets
        
        super().__init__(state_dim, action_dim, hidden_dim)
        
        print(f"📊 SAC Critic simplifié initialisé: {state_dim}+{action_dim}→1 (sans attention)")


# Fonction utilitaire pour créer les modèles
def create_sac_models(num_assets: int, device: torch.device = None):
    """
    Crée les modèles SAC simplifiés pour l'entraînement
    """
    device = device or torch.device('cpu')
    
    # Actor
    actor = ActorWithAttention(num_assets=num_assets)
    actor.to(device)
    
    # Critics (double critic pour stabilité)
    critic1 = CriticWithAttention(num_assets=num_assets) 
    critic2 = CriticWithAttention(num_assets=num_assets)
    critic1.to(device)
    critic2.to(device)
    
    # Target critics
    target_critic1 = CriticWithAttention(num_assets=num_assets)
    target_critic2 = CriticWithAttention(num_assets=num_assets)
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())
    target_critic1.to(device)
    target_critic2.to(device)
    
    return {
        'actor': actor,
        'critic1': critic1,
        'critic2': critic2, 
        'target_critic1': target_critic1,
        'target_critic2': target_critic2
    }