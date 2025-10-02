"""
Agent SAC avec replay buffer pour la gestion de portefeuille.
Implémente l'algorithme Soft Actor-Critic selon spec.md.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Tuple, Dict, Any, Optional
import copy

from config import Config
from models import create_sac_models

class ReplayBuffer:
    """Buffer de replay pour SAC"""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: torch.device = None):
        if device is None:
            device = Config.init_device()
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0
        
        # Pré-allouer les tensors pour l'efficacité
        self.states = torch.zeros(capacity, state_dim, device=self.device)
        self.actions = torch.zeros(capacity, action_dim, device=self.device)
        self.rewards = torch.zeros(capacity, 1, device=self.device)
        self.next_states = torch.zeros(capacity, state_dim, device=self.device)
        self.dones = torch.zeros(capacity, 1, dtype=torch.bool, device=self.device)
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, 
             next_state: np.ndarray, done: bool):
        """Ajoute une transition au buffer"""
        
        self.states[self.position] = torch.FloatTensor(state).to(self.device)
        self.actions[self.position] = torch.FloatTensor(action).to(self.device)
        self.rewards[self.position] = torch.FloatTensor([reward]).to(self.device)
        self.next_states[self.position] = torch.FloatTensor(next_state).to(self.device)
        self.dones[self.position] = torch.BoolTensor([done]).to(self.device)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Échantillonne un batch de transitions"""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return (
            self.states[indices],
            self.actions[indices], 
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self):
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        """Vérifie si le buffer contient assez d'échantillons"""
        return len(self) >= batch_size


class SACAgent:
    """
    Agent Soft Actor-Critic avec mécanisme d'attention pour la gestion de portefeuille.
    Implémente l'algorithme SAC complet avec entropy regularization.
    """
    
    def __init__(self, 
                 num_assets: int,
                 state_dim: int,
                 action_dim: int,
                 device: torch.device = None,
                 target_entropy: Optional[float] = None):
        
        if device is None:
            device = Config.init_device()
            
        self.num_assets = num_assets
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Entropy target pour l'algorithme SAC
        self.target_entropy = target_entropy or -action_dim
        
        # Initialiser les modèles
        # Créer les modèles SAC simplifiés
        models_dict = create_sac_models(num_assets, device)
        self.actor = models_dict['actor']
        self.critic1 = models_dict['critic1']
        self.critic2 = models_dict['critic2']
        self.target_critic1 = models_dict['target_critic1']
        self.target_critic2 = models_dict['target_critic2']
        
        # Temperature parameter pour SAC
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=Config.ACTOR_LR
        )
        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(), 
            lr=Config.CRITIC_LR
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(), 
            lr=Config.CRITIC_LR
        )
        self.alpha_optimizer = optim.Adam(
            [self.log_alpha], 
            lr=Config.ALPHA_LR
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=Config.REPLAY_BUFFER_SIZE,
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device
        )
        
        # Compteurs et statistiques
        self.training_step = 0
        self.actor_losses = deque(maxlen=1000)
        self.critic_losses = deque(maxlen=1000)
        self.alpha_losses = deque(maxlen=1000)
        self.q_values = deque(maxlen=1000)
        
        print(f"  Agent SAC initialisé:")
        print(f"   State dim: {state_dim}")
        print(f"   Action dim: {action_dim}")
        print(f"   Target entropy: {self.target_entropy:.2f}")
        print(f"   Device: {device}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Sélectionne une action selon la politique actuelle"""
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if training:
                # Mode entraînement: échantillonner depuis la politique
                action, _, _ = self.models.actor.sample(state_tensor)
            else:
                # Mode évaluation: utiliser la moyenne de la politique
                mean, _ = self.models.actor(state_tensor)
                action = torch.tanh(mean)
                action = (action + 1) / 2  # [-1, 1] -> [0, 1]
                action = F.softmax(action * 5, dim=1)  # Normaliser
        
        return action.cpu().numpy().flatten()
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool):
        """Stocke une transition dans le replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self, batch_size: int = Config.BATCH_SIZE) -> Dict[str, float]:
        """Met à jour l'agent SAC"""
        
        if not self.replay_buffer.is_ready(batch_size):
            return {}
        
        # Échantillonner un batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Mettre à jour les critics
        critic_loss = self._update_critics(states, actions, rewards, next_states, dones)
        
        # Mettre à jour l'actor et alpha
        actor_loss, alpha_loss = self._update_actor_and_alpha(states)
        
        # Soft update des targets
        self.models.soft_update_targets()
        
        # Mettre à jour les statistiques
        self.training_step += 1
        
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.models.alpha.item(),
            'training_step': self.training_step
        }
    
    def _update_critics(self, states: torch.Tensor, actions: torch.Tensor, 
                       rewards: torch.Tensor, next_states: torch.Tensor, 
                       dones: torch.Tensor) -> float:
        """Met à jour les réseaux critics"""
        
        with torch.no_grad():
            # Échantillonner les actions suivantes depuis la politique
            next_actions, next_log_probs, _ = self.models.actor.sample(next_states)
            
            # Calculer les Q-values target
            q1_target = self.models.critic1_target(next_states, next_actions)
            q2_target = self.models.critic2_target(next_states, next_actions)
            min_q_target = torch.min(q1_target, q2_target)
            
            # Ajouter l'entropy bonus
            next_q_value = min_q_target - self.models.alpha * next_log_probs
            
            # Calculer la target value
            target_q = rewards + (1 - dones.float()) * Config.GAMMA * next_q_value
        
        # Q-values actuelles
        current_q1 = self.models.critic1(states, actions)
        current_q2 = self.models.critic2(states, actions)
        
        # Losses des critics
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # Mise à jour critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.models.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()
        
        # Mise à jour critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.models.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()
        
        # Enregistrer les statistiques
        critic_loss = (critic1_loss + critic2_loss).item() / 2
        self.critic_losses.append(critic_loss)
        self.q_values.append(current_q1.mean().item())
        
        return critic_loss
    
    def _update_actor_and_alpha(self, states: torch.Tensor) -> Tuple[float, float]:
        """Met à jour l'actor et le paramètre alpha"""
        
        # Échantillonner des actions depuis la politique actuelle
        actions, log_probs, _ = self.models.actor.sample(states)
        
        # Q-values pour les nouvelles actions
        q1 = self.models.critic1(states, actions)
        q2 = self.models.critic2(states, actions)
        min_q = torch.min(q1, q2)
        
        # Loss de l'actor (maximiser Q - α * entropy)
        actor_loss = (self.models.alpha * log_probs - min_q).mean()
        
        # Mise à jour de l'actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.models.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Mise à jour d'alpha (temperature parameter)
        alpha_loss = -(self.models.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Enregistrer les statistiques
        self.actor_losses.append(actor_loss.item())
        self.alpha_losses.append(alpha_loss.item())
        
        return actor_loss.item(), alpha_loss.item()
    
    def get_training_stats(self) -> Dict[str, float]:
        """Retourne les statistiques d'entraînement"""
        stats = {}
        
        if self.actor_losses:
            stats['avg_actor_loss'] = np.mean(self.actor_losses)
        if self.critic_losses:
            stats['avg_critic_loss'] = np.mean(self.critic_losses)
        if self.alpha_losses:
            stats['avg_alpha_loss'] = np.mean(self.alpha_losses)
        if self.q_values:
            stats['avg_q_value'] = np.mean(self.q_values)
        
        stats['alpha'] = self.models.alpha.item()
        stats['buffer_size'] = len(self.replay_buffer)
        stats['training_steps'] = self.training_step
        
        return stats
    
    def save(self, filepath: str):
        """Sauvegarde l'agent complet"""
        save_dict = {
            'models_state': {
                'actor': self.models.actor.state_dict(),
                'critic1': self.models.critic1.state_dict(),
                'critic2': self.models.critic2.state_dict(),
                'critic1_target': self.models.critic1_target.state_dict(),
                'critic2_target': self.models.critic2_target.state_dict(),
                'log_alpha': self.models.log_alpha,
            },
            'optimizers_state': {
                'actor': self.actor_optimizer.state_dict(),
                'critic1': self.critic1_optimizer.state_dict(),
                'critic2': self.critic2_optimizer.state_dict(),
                'alpha': self.alpha_optimizer.state_dict(),
            },
            'training_step': self.training_step,
            'target_entropy': self.target_entropy,
        }
        
        torch.save(save_dict, filepath)
        print(f"  Agent sauvegardé dans {filepath}")
    
    def load(self, filepath: str):
        """Charge l'agent complet"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Charger les modèles
        self.models.actor.load_state_dict(checkpoint['models_state']['actor'])
        self.models.critic1.load_state_dict(checkpoint['models_state']['critic1'])
        self.models.critic2.load_state_dict(checkpoint['models_state']['critic2'])
        self.models.critic1_target.load_state_dict(checkpoint['models_state']['critic1_target'])
        self.models.critic2_target.load_state_dict(checkpoint['models_state']['critic2_target'])
        self.models.log_alpha = checkpoint['models_state']['log_alpha']
        
        # Charger les optimizers
        self.actor_optimizer.load_state_dict(checkpoint['optimizers_state']['actor'])
        self.critic1_optimizer.load_state_dict(checkpoint['optimizers_state']['critic1'])
        self.critic2_optimizer.load_state_dict(checkpoint['optimizers_state']['critic2'])
        self.alpha_optimizer.load_state_dict(checkpoint['optimizers_state']['alpha'])
        
        # Autres états
        self.training_step = checkpoint['training_step']
        self.target_entropy = checkpoint['target_entropy']
        
        print(f"  Agent chargé depuis {filepath}")


def test_agent():
    """Test de l'agent SAC"""
    print("🧪 Test de l'agent SAC...")
    
    # Paramètres de test
    num_assets = 6
    feature_dim = Config.FEATURE_DIM
    state_dim = num_assets * feature_dim + num_assets + 1 + num_assets  # 185 pour 8 assets
    action_dim = num_assets
    
    # Créer l'agent
    agent = SACAgent(
        num_assets=num_assets,
        state_dim=state_dim,
        action_dim=action_dim
    )
    
    # Simuler quelques transitions
    print("\n📝 Simulation de transitions...")
    for i in range(100):
        # État aléatoire
        state = np.random.randn(state_dim)
        
        # Sélectionner une action
        action = agent.select_action(state, training=True)
        
        # Transition simulée
        next_state = np.random.randn(state_dim)
        reward = np.random.randn()
        done = i == 99  # Dernier step
        
        # Stocker la transition
        agent.store_transition(state, action, reward, next_state, done)
        
        # Mise à jour après quelques transitions
        if i >= 32:  # Attendre d'avoir assez d'échantillons
            update_info = agent.update(batch_size=16)
            if i % 20 == 0 and update_info:
                print(f"  Step {i}: Actor loss = {update_info.get('actor_loss', 0):.4f}, "
                      f"Critic loss = {update_info.get('critic_loss', 0):.4f}, "
                      f"Alpha = {update_info.get('alpha', 0):.4f}")
    
    # Statistiques finales
    stats = agent.get_training_stats()
    print(f"\n📊 Statistiques finales:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Test de sauvegarde/chargement
    print("\n  Test sauvegarde/chargement...")
    save_path = "test_agent.pth"
    agent.save(save_path)
    
    # Créer un nouvel agent et charger
    new_agent = SACAgent(num_assets, state_dim, action_dim)
    new_agent.load(save_path)
    
    # Vérifier que les actions sont similaires
    test_state = np.random.randn(state_dim)
    action1 = agent.select_action(test_state, training=False)
    action2 = new_agent.select_action(test_state, training=False)
    
    similarity = np.corrcoef(action1, action2)[0, 1]
    print(f"  Similarité des actions: {similarity:.4f}")
    
    # Nettoyer
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
    
    print("\n  Test de l'agent terminé avec succès!")


if __name__ == "__main__":
    test_agent()
