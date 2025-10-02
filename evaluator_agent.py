"""
Agent SAC simplifié pour l'évaluation des modèles entraînés
"""

import torch
import numpy as np
from models import create_sac_models

class SACAgentEvaluator:
    """Agent SAC simplifié pour évaluation seulement"""
    
    def __init__(self, num_assets: int, state_dim: int, action_dim: int, device: torch.device = None):
        if device is None:
            device = torch.device("cpu")
            
        self.num_assets = num_assets
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Créer les modèles
        models_dict = create_sac_models(num_assets, device)
        self.actor = models_dict['actor']
        self.critic1 = models_dict['critic1']
        self.critic2 = models_dict['critic2']
        
    def select_action(self, state, evaluate=True):
        """Sélectionne une action"""
        with torch.no_grad():
            if evaluate:
                # Mode évaluation: utiliser la moyenne
                mean, _ = self.actor(state)
                action = torch.softmax(mean, dim=-1)  # Normalisation
            else:
                # Mode exploration: échantillonner
                action, _, _ = self.actor.sample(state)
                
            return action.cpu().numpy().flatten()
    
    def load_state_dict(self, state_dict):
        """Charge les poids du modèle"""
        if 'actor' in state_dict:
            self.actor.load_state_dict(state_dict['actor'])
        if 'critic1' in state_dict:
            self.critic1.load_state_dict(state_dict['critic1'])
        if 'critic2' in state_dict:
            self.critic2.load_state_dict(state_dict['critic2'])