"""
Script autonome d'entraînement SAC pour Kaggle.
Inclut tous les modules nécessaires pour fonctionner sans imports externes.
"""

import os
import sys
import time
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting Kaggle Training Script...")
print(f"Working directory: {os.getcwd()}")
print(f"CUDA available: {torch.cuda.is_available()}")

# =============================================================================
# CONFIG MODULE (embedded)
# =============================================================================

class Config:
    """Configuration centralisée pour l'agent SAC et l'environnement."""
    
    # Configuration de l'environnement
    LOOKBACK_WINDOW = 30
    TRANSACTION_COST = 0.002  # 0.2%
    INITIAL_BALANCE = 100000
    
    # Paramètres SAC
    LEARNING_RATE = 3e-4
    GAMMA = 0.99
    TAU = 0.005
    BATCH_SIZE = 256
    BUFFER_SIZE = 1000000
    ALPHA = 0.2  # Temperature parameter
    
    # Réseaux de neurones
    HIDDEN_SIZE = 256
    
    # Entraînement
    MAX_EPISODES = 1000
    EPISODE_LENGTH = 252  # 1 année de trading
    UPDATE_FREQUENCY = 1
    EVAL_FREQUENCY = 50
    SAVE_FREQUENCY = 100
    
    # Gestion des modèles
    MODEL_DIR = "models"
    RESULTS_DIR = "results"
    
    @classmethod
    def get_device(cls) -> torch.device:
        """Retourne le device PyTorch approprié."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"⚡ Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print("💻 Using CPU")
        return device

# =============================================================================
# DATA PROCESSING MODULE (embedded)
# =============================================================================

class DataHandler:
    """Gestionnaire de données pour l'environnement de trading."""
    
    def __init__(self, data_path: str = "datas/all_datas.xlsx"):
        self.data_path = data_path
        self.data = None
        self.returns = None
        
    def load_data(self) -> pd.DataFrame:
        """Charge les données depuis le fichier Excel."""
        try:
            self.data = pd.read_excel(self.data_path, index_col=0, parse_dates=True)
            print(f"✅ Data loaded: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            # Fallback: generate synthetic data
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Génère des données synthétiques pour les tests."""
        print("🔧 Generating synthetic data...")
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        n_assets = 10
        
        # Génération de rendements avec corrélations
        np.random.seed(42)
        returns = np.random.multivariate_normal(
            mean=[0.0005] * n_assets,
            cov=np.eye(n_assets) * 0.01 + np.ones((n_assets, n_assets)) * 0.002,
            size=len(dates)
        )
        
        # Conversion en prix
        prices = np.cumprod(1 + returns, axis=0) * 100
        
        columns = [f'Asset_{i}' for i in range(n_assets)]
        data = pd.DataFrame(prices, index=dates, columns=columns)
        
        self.data = data
        print(f"✅ Synthetic data generated: {data.shape}")
        return data
    
    def calculate_returns(self) -> pd.DataFrame:
        """Calcule les rendements des actifs."""
        if self.data is None:
            self.load_data()
        
        self.returns = self.data.pct_change().dropna()
        return self.returns
    
    def get_features(self, lookback: int = 30) -> np.ndarray:
        """Extrait les caractéristiques pour l'agent."""
        if self.returns is None:
            self.calculate_returns()
        
        # Features: returns, volatility, momentum
        returns_features = self.returns.rolling(lookback).mean()
        volatility_features = self.returns.rolling(lookback).std()
        momentum_features = self.data.pct_change(lookback)
        
        features = pd.concat([returns_features, volatility_features, momentum_features], axis=1)
        return features.dropna().values

class FeatureProcessor:
    """Processeur de caractéristiques avancées."""
    
    @staticmethod
    def technical_indicators(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Calcule les indicateurs techniques."""
        indicators = pd.DataFrame(index=prices.index)
        
        for col in prices.columns:
            # Moving averages
            indicators[f'{col}_MA'] = prices[col].rolling(window).mean()
            indicators[f'{col}_RSI'] = FeatureProcessor.rsi(prices[col], window)
            
        return indicators
    
    @staticmethod
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """Calcule le RSI."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

# =============================================================================
# ENVIRONMENT MODULE (embedded)
# =============================================================================

class PortfolioEnv:
    """Environnement de gestion de portefeuille pour l'RL."""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000, 
                 transaction_cost: float = 0.002, lookback_window: int = 30):
        self.data = data
        self.returns = data.pct_change().dropna()
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        
        self.n_assets = len(data.columns)
        self.reset()
        
    def reset(self) -> Tuple[np.ndarray, dict]:
        """Reset l'environnement."""
        self.current_step = self.lookback_window
        self.portfolio_value = self.initial_balance
        self.cash = self.initial_balance
        self.holdings = np.zeros(self.n_assets)
        self.portfolio_weights = np.zeros(self.n_assets)
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Retourne l'observation actuelle."""
        if self.current_step < self.lookback_window:
            return np.zeros(self.n_assets * 3)
        
        # Features: returns, volatility, current weights
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        
        returns_window = self.returns.iloc[start_idx:end_idx].mean().values
        volatility_window = self.returns.iloc[start_idx:end_idx].std().values
        
        observation = np.concatenate([
            returns_window,
            volatility_window, 
            self.portfolio_weights
        ])
        
        return observation.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute une action dans l'environnement."""
        # Normalize action to portfolio weights
        action = np.clip(action, 0, 1)
        if action.sum() > 0:
            action = action / action.sum()
        
        # Calculate costs of rebalancing
        weight_changes = np.abs(action - self.portfolio_weights)
        transaction_costs = np.sum(weight_changes) * self.transaction_cost * self.portfolio_value
        
        # Update portfolio
        if self.current_step < len(self.data):
            current_returns = self.returns.iloc[self.current_step]
            
            # Portfolio return
            portfolio_return = np.sum(self.portfolio_weights * current_returns)
            self.portfolio_value *= (1 + portfolio_return)
            self.portfolio_value -= transaction_costs
            
            # Update weights
            self.portfolio_weights = action
            
            # Calculate reward
            reward = portfolio_return - (transaction_costs / self.portfolio_value)
        else:
            reward = 0
        
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        truncated = False
        
        info = {
            'portfolio_value': self.portfolio_value,
            'transaction_costs': transaction_costs,
            'portfolio_return': portfolio_return if 'portfolio_return' in locals() else 0
        }
        
        return self._get_observation(), reward, done, truncated, info

# =============================================================================
# NEURAL NETWORKS MODULE (embedded) 
# =============================================================================

class PolicyNetwork(nn.Module):
    """Réseau de politique pour SAC."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        self.log_std_min = -20
        self.log_std_max = 2
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.softmax(x_t, dim=-1)
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(action + 1e-8)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob

class QNetwork(nn.Module):
    """Réseau Q pour SAC."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# =============================================================================
# SAC AGENT MODULE (embedded)
# =============================================================================

class ReplayBuffer:
    """Buffer de replay pour SAC."""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros((capacity, action_dim))
        self.rewards = np.zeros((capacity, 1))
        self.next_states = np.zeros((capacity, state_dim))
        self.dones = np.zeros((capacity, 1))
        
    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size: int, device: torch.device):
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.states[indices]).to(device),
            torch.FloatTensor(self.actions[indices]).to(device),
            torch.FloatTensor(self.rewards[indices]).to(device),
            torch.FloatTensor(self.next_states[indices]).to(device),
            torch.FloatTensor(self.dones[indices]).to(device)
        )
    
    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size

class SACAgent:
    """Agent SAC pour la gestion de portefeuille."""
    
    def __init__(self, state_dim: int, action_dim: int, device: torch.device):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.q1 = QNetwork(state_dim, action_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim).to(device)
        self.q1_target = QNetwork(state_dim, action_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim).to(device)
        
        # Copy parameters to target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=Config.LEARNING_RATE)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=Config.LEARNING_RATE)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=Config.LEARNING_RATE)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(Config.BUFFER_SIZE, state_dim, action_dim)
        
        # Temperature parameter
        self.alpha = Config.ALPHA
        
        # Training stats
        self.training_stats = {
            'policy_loss': [],
            'q1_loss': [],
            'q2_loss': [],
            'alpha': []
        }
        
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Sélectionne une action."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if training:
            action, _ = self.policy.sample(state_tensor)
        else:
            mean, _ = self.policy(state_tensor)
            action = torch.softmax(mean, dim=-1)
            
        return action.cpu().data.numpy().flatten()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Stocke une transition dans le buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
        
    def update(self) -> Dict[str, float]:
        """Met à jour l'agent."""
        if not self.replay_buffer.is_ready(Config.BATCH_SIZE):
            return {}
            
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            Config.BATCH_SIZE, self.device
        )
        
        # Update Q-networks
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + Config.GAMMA * (1 - dones) * q_next
        
        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Update policy
        new_actions, log_probs = self.policy.sample(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_probs - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update target networks
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(Config.TAU * param.data + (1 - Config.TAU) * target_param.data)
            
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(Config.TAU * param.data + (1 - Config.TAU) * target_param.data)
        
        # Store stats
        self.training_stats['policy_loss'].append(policy_loss.item())
        self.training_stats['q1_loss'].append(q1_loss.item())
        self.training_stats['q2_loss'].append(q2_loss.item())
        self.training_stats['alpha'].append(self.alpha)
        
        return {
            'policy_loss': policy_loss.item(),
            'q1_loss': q1_loss.item(), 
            'q2_loss': q2_loss.item(),
            'alpha': self.alpha
        }
    
    def save(self, path: str):
        """Sauvegarde le modèle."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
            'training_stats': self.training_stats
        }, path)
        
    def load(self, path: str):
        """Charge le modèle."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        
    def get_training_stats(self) -> Dict:
        """Retourne les statistiques d'entraînement."""
        return self.training_stats

# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def main():
    """Fonction principale d'entraînement."""
    print("🎯 Starting RL Portfolio Optimizer Training on Kaggle")
    
    # Configuration
    device = Config.get_device()
    
    # Load and prepare data
    data_handler = DataHandler()
    data = data_handler.load_data()
    
    if data is None or len(data) < 100:
        print("❌ Insufficient data, using synthetic data")
        data = data_handler._generate_synthetic_data()
    
    # Create environment
    env = PortfolioEnv(data, Config.INITIAL_BALANCE, Config.TRANSACTION_COST, Config.LOOKBACK_WINDOW)
    
    # Get dimensions
    state_dim = env._get_observation().shape[0]
    action_dim = env.n_assets
    
    print(f"📊 Environment: {state_dim} states, {action_dim} actions")
    
    # Create agent
    agent = SACAgent(state_dim, action_dim, device)
    
    # Training parameters (simplified for Kaggle)
    num_episodes = 10  # Limited for Kaggle execution time
    
    print(f"🚀 Starting training for {num_episodes} episodes...")
    
    # Training loop
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        step = 0
        
        while True:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Environment step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent
            if agent.replay_buffer.is_ready(Config.BATCH_SIZE):
                update_info = agent.update()
                
            episode_reward += reward
            state = next_state
            step += 1
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        
        print(f"Episode {episode+1}/{num_episodes}: "
              f"Reward={episode_reward:.4f}, "
              f"Portfolio Value={info.get('portfolio_value', 0):.2f}, "
              f"Steps={step}")
    
    # Save results
    print(f"\n🎉 Training completed!")
    print(f"📊 Average reward: {np.mean(total_rewards):.4f}")
    print(f"📈 Best reward: {np.max(total_rewards):.4f}")
    print(f"💼 Final portfolio value: {info.get('portfolio_value', Config.INITIAL_BALANCE):.2f}")
    
    # Save model
    try:
        os.makedirs('models', exist_ok=True)
        model_path = 'models/sac_portfolio_agent_kaggle.pth'
        agent.save(model_path)
        print(f"💾 Model saved to {model_path}")
    except Exception as e:
        print(f"⚠️ Could not save model: {e}")
    
    # Save training summary
    try:
        summary = {
            'episodes': num_episodes,
            'total_rewards': total_rewards,
            'average_reward': float(np.mean(total_rewards)),
            'best_reward': float(np.max(total_rewards)),
            'final_portfolio_value': float(info.get('portfolio_value', Config.INITIAL_BALANCE)),
            'training_stats': agent.get_training_stats()
        }
        
        with open('training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"📄 Training summary saved to training_summary.json")
        
    except Exception as e:
        print(f"⚠️ Could not save summary: {e}")
    
    print("✅ Kaggle training script completed successfully!")

if __name__ == "__main__":
    main()