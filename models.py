"""
Modèles de réseaux de neurones pour l'agent SAC avec mécanisme d'attention.
Implémente ActorWithAttention et CriticWithAttention selon spec.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any
import math

from config import Config

class PositionalEncoding(nn.Module):
    """Encodage positionnel pour le mécanisme d'attention"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SelfAttentionModule(nn.Module):
    """Module d'auto-attention pour analyser les relations entre assets"""
    
    def __init__(self, feature_dim: int, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Projection des features vers la dimension cachée
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        
        # Couche d'auto-attention multi-têtes
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Normalisation et feed-forward
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Encodage positionnel
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de l'attention.
        Args:
            x: Tensor de shape (batch_size, num_assets, feature_dim)
        Returns:
            Tensor de shape (batch_size, num_assets, hidden_dim)
        """
        batch_size, num_assets, _ = x.shape
        
        # Projection vers la dimension cachée
        x = self.input_projection(x)
        
        # Ajouter l'encodage positionnel
        x = self.pos_encoding(x)
        
        # Auto-attention
        attn_output, attention_weights = self.multi_head_attention(x, x, x)
        
        # Connexion résiduelle + normalisation
        x = self.layer_norm1(x + attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        
        # Connexion résiduelle + normalisation
        x = self.layer_norm2(x + ff_output)
        
        return x


class ActorWithAttention(nn.Module):
    """
    Réseau Actor avec mécanisme d'attention pour la politique SAC.
    Produit une distribution de probabilité sur les allocations de portefeuille.
    """
    
    def __init__(self, 
                 num_assets: int,
                 feature_dim: int = Config.FEATURE_DIM,
                 hidden_dim: int = Config.HIDDEN_DIM,
                 attention_heads: int = Config.ATTENTION_HEADS):
        super().__init__()
        
        self.num_assets = num_assets
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Module d'attention pour analyser les relations entre assets
        self.attention = SelfAttentionModule(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_heads=attention_heads
        )
        
        # Agrégation des features d'attention
        self.aggregation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Traitement de l'état du portefeuille
        portfolio_state_dim = num_assets + 1 + num_assets  # weights + cash + holdings
        self.portfolio_encoder = nn.Sequential(
            nn.Linear(portfolio_state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Réseau de la politique
        combined_dim = (hidden_dim // 2) * num_assets + hidden_dim // 2
        self.policy_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_assets * 2)  # Mean et log_std pour chaque asset
        )
        
        # Limites pour log_std
        self.log_std_min = -20
        self.log_std_max = 2
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass de l'actor.
        Args:
            state: État complet incluant features et état du portefeuille
        Returns:
            mean, log_std des actions
        """
        batch_size = state.size(0)
        
        # Découper l'état
        features_flat = state[:, :self.num_assets * self.feature_dim]
        portfolio_state = state[:, self.num_assets * self.feature_dim:]
        
        # Reshaper les features: (batch, num_assets, feature_dim)
        features = features_flat.view(batch_size, self.num_assets, self.feature_dim)
        
        # Appliquer l'attention
        attention_output = self.attention(features)  # (batch, num_assets, hidden_dim)
        
        # Agréger les features d'attention pour chaque asset
        aggregated_features = self.aggregation(attention_output)  # (batch, num_assets, hidden_dim//2)
        aggregated_features_flat = aggregated_features.view(batch_size, -1)
        
        # Encoder l'état du portefeuille
        portfolio_encoded = self.portfolio_encoder(portfolio_state)  # (batch, hidden_dim//2)
        
        # Combiner les features d'attention et l'état du portefeuille
        combined = torch.cat([aggregated_features_flat, portfolio_encoded], dim=1)
        
        # Passer par le réseau de politique
        policy_output = self.policy_net(combined)
        
        # Séparer mean et log_std
        mean = policy_output[:, :self.num_assets]
        log_std = policy_output[:, self.num_assets:]
        
        # Contraindre log_std
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Échantillonne une action à partir de la politique.
        Returns:
            action, log_prob, mean
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Distribution normale
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Appliquer tanh pour contraindre à [-1, 1]
        action_raw = torch.tanh(x_t)
        
        # Calculer log_prob avec correction de Jacobien
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action_raw.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        # Transformer vers [0, 1] et normaliser
        action = (action_raw + 1) / 2  # [-1, 1] -> [0, 1]
        action = F.softmax(action * 5, dim=1)  # Softmax avec température
        
        return action, log_prob, torch.tanh(mean)


class CriticWithAttention(nn.Module):
    """
    Réseau Critic avec mécanisme d'attention pour les Q-values SAC.
    Évalue la valeur état-action.
    """
    
    def __init__(self, 
                 num_assets: int,
                 feature_dim: int = Config.FEATURE_DIM,
                 hidden_dim: int = Config.HIDDEN_DIM,
                 attention_heads: int = Config.ATTENTION_HEADS):
        super().__init__()
        
        self.num_assets = num_assets
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Module d'attention pour analyser les relations entre assets
        self.attention = SelfAttentionModule(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_heads=attention_heads
        )
        
        # Agrégation des features d'attention
        self.features_aggregation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Traitement de l'état du portefeuille
        portfolio_state_dim = num_assets + 1 + num_assets  # weights + cash + holdings
        self.portfolio_encoder = nn.Sequential(
            nn.Linear(portfolio_state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Traitement des actions
        self.action_encoder = nn.Sequential(
            nn.Linear(num_assets, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Réseau Q-value
        combined_dim = (hidden_dim // 2) * num_assets + hidden_dim // 2 + hidden_dim // 4
        self.q_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass du critic.
        Args:
            state: État complet
            action: Action (allocation de portefeuille)
        Returns:
            Q-value
        """
        batch_size = state.size(0)
        
        # Découper l'état
        features_flat = state[:, :self.num_assets * self.feature_dim]
        portfolio_state = state[:, self.num_assets * self.feature_dim:]
        
        # Reshaper les features: (batch, num_assets, feature_dim)
        features = features_flat.view(batch_size, self.num_assets, self.feature_dim)
        
        # Appliquer l'attention
        attention_output = self.attention(features)  # (batch, num_assets, hidden_dim)
        
        # Agréger les features d'attention
        aggregated_features = self.features_aggregation(attention_output)
        aggregated_features_flat = aggregated_features.view(batch_size, -1)
        
        # Encoder l'état du portefeuille
        portfolio_encoded = self.portfolio_encoder(portfolio_state)
        
        # Encoder l'action
        action_encoded = self.action_encoder(action)
        
        # Combiner tous les éléments
        combined = torch.cat([aggregated_features_flat, portfolio_encoded, action_encoded], dim=1)
        
        # Calculer la Q-value
        q_value = self.q_net(combined)
        
        return q_value

from config import Config

class SACModels:
    """Conteneur pour tous les modèles SAC"""
    
    def __init__(self, num_assets: int, device: torch.device = None):
        if device is None:
            device = Config.init_device()
        self.num_assets = num_assets
        self.device = device
        
        # Créer les modèles
        self.actor = ActorWithAttention(num_assets).to(device)
        self.critic1 = CriticWithAttention(num_assets).to(device)
        self.critic2 = CriticWithAttention(num_assets).to(device)
        
        # Créer les targets (copies des critics)
        self.critic1_target = CriticWithAttention(num_assets).to(device)
        self.critic2_target = CriticWithAttention(num_assets).to(device)
        
        # Initialiser les targets avec les mêmes poids
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Geler les paramètres des targets
        for param in self.critic1_target.parameters():
            param.requires_grad = False
        for param in self.critic2_target.parameters():
            param.requires_grad = False
        
        # Paramètre d'entropie (apprenable)
        self.log_alpha = torch.tensor(
            np.log(Config.INITIAL_ALPHA), 
            dtype=torch.float32, 
            device=device, 
            requires_grad=True
        )

        
        print(f"✅ Modèles SAC initialisés avec {num_assets} assets")
        print(f"   Actor parameters: {sum(p.numel() for p in self.actor.parameters()):,}")
        print(f"   Critic parameters: {sum(p.numel() for p in self.critic1.parameters()):,}")
    
    @property
    def alpha(self):
        """Coefficient d'entropie"""
        return self.log_alpha.exp()
    
    def soft_update_targets(self, tau: float = Config.TAU):
        """Mise à jour douce des réseaux targets"""
        for target_param, param in zip(self.critic1_target.parameters(), 
                                     self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for target_param, param in zip(self.critic2_target.parameters(), 
                                     self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save_models(self, path: str):
        """Sauvegarde tous les modèles"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'log_alpha': self.log_alpha,
        }, path)
        print(f"✅ Modèles sauvegardés dans {path}")
    
    def load_models(self, path: str):
        """Charge tous les modèles"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.log_alpha = checkpoint['log_alpha']
        
        print(f"✅ Modèles chargés depuis {path}")


def test_models():
    """Test des modèles avec attention"""
    print("🧪 Test des modèles SAC avec attention...")
    
    # Paramètres de test
    batch_size = 4
    num_assets = 8
    feature_dim = Config.FEATURE_DIM  # 21
    
    # Calcul de la dimension d'observation
    obs_dim = num_assets * feature_dim + num_assets + 1 + num_assets
    
    # Créer des données de test
    state = torch.randn(batch_size, obs_dim)
    action = torch.rand(batch_size, num_assets)
    action = F.softmax(action, dim=1)  # Normaliser les actions
    
    print(f"State shape: {state.shape}")
    print(f"Action shape: {action.shape}")
    
    # Tester les modèles
    models = SACModels(num_assets)
    
    # Test Actor
    print("\n🎭 Test Actor:")
    mean, log_std = models.actor(state)
    print(f"  Mean shape: {mean.shape}")
    print(f"  Log_std shape: {log_std.shape}")
    
    # Test sampling
    sampled_action, log_prob, _ = models.actor.sample(state)
    print(f"  Sampled action shape: {sampled_action.shape}")
    print(f"  Log prob shape: {log_prob.shape}")
    print(f"  Action sum: {sampled_action.sum(dim=1)}")  # Doit être ~1
    
    # Test Critics
    print("\n🎯 Test Critics:")
    q1 = models.critic1(state, action)
    q2 = models.critic2(state, action)
    print(f"  Q1 shape: {q1.shape}")
    print(f"  Q2 shape: {q2.shape}")
    
    # Test targets
    with torch.no_grad():
        q1_target = models.critic1_target(state, action)
        q2_target = models.critic2_target(state, action)
        print(f"  Q1 target shape: {q1_target.shape}")
        print(f"  Q2 target shape: {q2_target.shape}")
    
    # Test soft update
    print("\n🔄 Test soft update:")
    old_param = models.critic1_target.q_net[0].weight.clone()
    models.soft_update_targets(tau=0.1)
    new_param = models.critic1_target.q_net[0].weight
    param_changed = not torch.equal(old_param, new_param)
    print(f"  Parameters changed: {param_changed}")
    
    print("\n✅ Test des modèles terminé avec succès!")


if __name__ == "__main__":
    test_models()
