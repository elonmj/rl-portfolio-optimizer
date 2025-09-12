"""
Script d'entraînement de l'agent SAC pour la gestion de portefeuille.
Implémente la boucle d'entraînement et validation selon spec.md.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import Config
from data_processing import DataHandler, FeatureProcessor
from environment import PortfolioEnv
from agent import SACAgent

class TrainingLogger:
    """Logger pour l'entraînement"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics = {
            'episode': [],
            'total_reward': [],
            'portfolio_value': [],
            'total_return': [],
            'sharpe_ratio': [],
            'max_drawdown': [],
            'num_trades': [],
            'actor_loss': [],
            'critic_loss': [],
            'alpha': [],
            'cvar': []
        }
        
    def log_episode(self, episode: int, metrics: Dict):
        """Enregistre les métriques d'un épisode"""
        self.metrics['episode'].append(episode)
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def save_metrics(self):
        """Sauvegarde les métriques en CSV"""
        df = pd.DataFrame(self.metrics)
        df.to_csv(os.path.join(self.log_dir, 'training_metrics.csv'), index=False)
    
    def plot_training_curves(self):
        """Génère les courbes d'entraînement"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Rewards
        axes[0, 0].plot(self.metrics['episode'], self.metrics['total_reward'])
        axes[0, 0].set_title('Total Reward')
        axes[0, 0].set_xlabel('Episode')
        
        # Portfolio Value
        axes[0, 1].plot(self.metrics['episode'], self.metrics['portfolio_value'])
        axes[0, 1].set_title('Portfolio Value')
        axes[0, 1].set_xlabel('Episode')
        
        # Total Return
        axes[0, 2].plot(self.metrics['episode'], self.metrics['total_return'])
        axes[0, 2].set_title('Total Return')
        axes[0, 2].set_xlabel('Episode')
        
        # Actor Loss
        if any(x is not None for x in self.metrics['actor_loss']):
            valid_losses = [(i, loss) for i, loss in enumerate(self.metrics['actor_loss']) if loss is not None]
            if valid_losses:
                episodes, losses = zip(*valid_losses)
                axes[1, 0].plot(episodes, losses)
        axes[1, 0].set_title('Actor Loss')
        axes[1, 0].set_xlabel('Episode')
        
        # Critic Loss
        if any(x is not None for x in self.metrics['critic_loss']):
            valid_losses = [(i, loss) for i, loss in enumerate(self.metrics['critic_loss']) if loss is not None]
            if valid_losses:
                episodes, losses = zip(*valid_losses)
                axes[1, 1].plot(episodes, losses)
        axes[1, 1].set_title('Critic Loss')
        axes[1, 1].set_xlabel('Episode')
        
        # Alpha
        if any(x is not None for x in self.metrics['alpha']):
            valid_alphas = [(i, alpha) for i, alpha in enumerate(self.metrics['alpha']) if alpha is not None]
            if valid_alphas:
                episodes, alphas = zip(*valid_alphas)
                axes[1, 2].plot(episodes, alphas)
        axes[1, 2].set_title('Alpha (Entropy Coefficient)')
        axes[1, 2].set_xlabel('Episode')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_curves.png'))
        plt.close()


class PortfolioTrainer:
    """Classe principale pour l'entraînement de l'agent SAC"""
    
    def __init__(self, config_overrides: Dict = None):
        self.config = Config()
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(self.config, key, value)
        
        # Initialiser les composants
        self.data_handler = DataHandler()
        self.feature_processor = FeatureProcessor(self.data_handler)
        
        # Créer les répertoires nécessaires
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        
        # Logger
        self.logger = TrainingLogger(Config.LOG_DIR)
        
        # Variables d'état
        self.train_env = None
        self.val_env = None
        self.agent = None
        self.best_val_return = -np.inf
        
        print("🚀 Trainer initialisé")
    
    def setup_environments(self):
        """Configure les environnements d'entraînement et de validation"""
        print("🏗️ Configuration des environnements...")
        
        # Charger les données
        self.data_handler.load_all_data()
        
        # Obtenir les tickers valides pour l'entraînement
        train_tickers = self.data_handler.get_available_tickers_for_period(
            Config.TRAIN_START, Config.TRAIN_END, min_observations=200
        )
        
        # Limiter le nombre de tickers pour l'entraînement
        max_tickers = min(len(train_tickers), 15)  # Maximum 15 tickers
        self.train_tickers = train_tickers[:max_tickers]
        
        print(f"   Tickers sélectionnés pour l'entraînement: {len(self.train_tickers)}")
        print(f"   Tickers: {self.train_tickers}")
        
        # Créer l'environnement d'entraînement
        self.train_env = PortfolioEnv(
            tickers=self.train_tickers,
            start_date=Config.TRAIN_START,
            end_date=Config.TRAIN_END,
            data_handler=self.data_handler,
            feature_processor=self.feature_processor
        )
        
        # Créer l'environnement de validation
        val_tickers = self.data_handler.get_available_tickers_for_period(
            Config.VALIDATION_START, Config.VALIDATION_END, min_observations=100
        )
        # Utiliser les mêmes tickers que l'entraînement si possible
        val_tickers = [t for t in self.train_tickers if t in val_tickers]
        
        if len(val_tickers) >= 5:  # Au moins 5 tickers pour la validation
            self.val_env = PortfolioEnv(
                tickers=val_tickers,
                start_date=Config.VALIDATION_START,
                end_date=Config.VALIDATION_END,
                data_handler=self.data_handler,
                feature_processor=self.feature_processor
            )
            print(f"   Validation configurée avec {len(val_tickers)} tickers")
        else:
            print("   ⚠️ Pas assez de tickers pour la validation")
            self.val_env = None
    
    def setup_agent(self):
        """Configure l'agent SAC"""
        print("🤖 Configuration de l'agent SAC...")
        
        if self.train_env is None:
            raise ValueError("Environnement d'entraînement non configuré")
        
        # Dimensions
        num_assets = len(self.train_tickers)
        state_dim = self.train_env.observation_space.shape[0]
        action_dim = self.train_env.action_space.shape[0]
        
        # Créer l'agent
        device = "cuda" if torch.cuda.is_available() and Config.DEVICE == "cuda" else "cpu"
        print(f"   Utilisation du device: {device}")
        
        self.agent = SACAgent(
            num_assets=num_assets,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device
        )
    
    def train_episode(self, episode: int) -> Dict:
        """Entraîne l'agent sur un épisode"""
        state, _ = self.train_env.reset()
        episode_reward = 0
        episode_steps = 0
        losses = []
        
        while True:
            # Sélectionner une action
            action = self.agent.select_action(state, training=True)
            
            # Exécuter l'action
            next_state, reward, terminated, truncated, info = self.train_env.step(action)
            done = terminated or truncated
            
            # Stocker la transition
            self.agent.store_transition(state, action, reward, next_state, done)
            
            # Mise à jour de l'agent
            if self.agent.replay_buffer.is_ready(Config.BATCH_SIZE):
                update_info = self.agent.update()
                if update_info:
                    losses.append(update_info)
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if done:
                break
        
        # Calculer les métriques de l'épisode
        portfolio_value = info.get('portfolio_value', Config.INITIAL_CASH)
        total_return = info.get('total_return', 0)
        
        metrics = {
            'total_reward': episode_reward,
            'portfolio_value': portfolio_value,
            'total_return': total_return,
            'episode_steps': episode_steps,
            'num_trades': info.get('num_active_positions', 0)
        }
        
        # Ajouter les métriques d'entraînement
        if losses:
            avg_losses = {
                key: np.mean([loss[key] for loss in losses if key in loss])
                for key in ['actor_loss', 'critic_loss', 'alpha']
            }
            metrics.update(avg_losses)
        
        return metrics
    
    def validate_agent(self) -> Dict:
        """Évalue l'agent sur l'environnement de validation"""
        if self.val_env is None:
            return {}
        
        state, _ = self.val_env.reset()
        episode_reward = 0
        portfolio_values = [Config.INITIAL_CASH]
        
        while True:
            # Action en mode évaluation (déterministe)
            action = self.agent.select_action(state, training=False)
            
            next_state, reward, terminated, truncated, info = self.val_env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            portfolio_values.append(info.get('portfolio_value', Config.INITIAL_CASH))
            state = next_state
            
            if done:
                break
        
        # Calculer les métriques de performance
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        metrics = {
            'val_total_reward': episode_reward,
            'val_portfolio_value': portfolio_values[-1],
            'val_total_return': (portfolio_values[-1] - Config.INITIAL_CASH) / Config.INITIAL_CASH,
            'val_sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'val_max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'val_cvar': info.get('cvar', 0)
        }
        
        return metrics
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calcule le ratio de Sharpe"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        
        annual_return = np.mean(returns) * 252  # 252 jours de trading par an
        annual_volatility = np.std(returns) * np.sqrt(252)
        
        return (annual_return - risk_free_rate) / annual_volatility
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calcule le maximum drawdown"""
        if len(portfolio_values) < 2:
            return 0
        
        values = np.array(portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        
        return np.max(drawdown)
    
    def train(self, num_episodes: int = Config.MAX_EPISODES):
        """Boucle principale d'entraînement"""
        print(f"🏋️ Démarrage de l'entraînement pour {num_episodes} épisodes")
        
        # Configuration
        self.setup_environments()
        self.setup_agent()
        
        start_time = time.time()
        
        for episode in tqdm(range(num_episodes), desc="Entraînement"):
            # Entraîner un épisode
            train_metrics = self.train_episode(episode)
            
            # Validation périodique
            val_metrics = {}
            if episode % Config.EVAL_FREQUENCY == 0:
                val_metrics = self.validate_agent()
                
                # Sauvegarder le meilleur modèle
                val_return = val_metrics.get('val_total_return', -np.inf)
                if val_return > self.best_val_return:
                    self.best_val_return = val_return
                    self.agent.save(os.path.join(Config.MODEL_DIR, 'best_model.pth'))
            
            # Logger les métriques
            all_metrics = {**train_metrics, **val_metrics}
            self.logger.log_episode(episode, all_metrics)
            
            # Affichage périodique
            if episode % 10 == 0:
                train_return = train_metrics.get('total_return', 0)
                val_return = val_metrics.get('val_total_return', 0)
                print(f"\nÉpisode {episode}:")
                print(f"  Train Return: {train_return:.2%}")
                if val_metrics:
                    print(f"  Val Return: {val_return:.2%}")
                    print(f"  Val Sharpe: {val_metrics.get('val_sharpe_ratio', 0):.2f}")
            
            # Sauvegarde périodique
            if episode % Config.SAVE_FREQUENCY == 0 and episode > 0:
                self.agent.save(os.path.join(Config.MODEL_DIR, f'model_episode_{episode}.pth'))
                self.logger.save_metrics()
                self.logger.plot_training_curves()
        
        # Fin de l'entraînement
        total_time = time.time() - start_time
        print(f"\n✅ Entraînement terminé en {total_time/60:.1f} minutes")
        print(f"   Meilleur retour validation: {self.best_val_return:.2%}")
        
        # Sauvegarde finale
        self.agent.save(os.path.join(Config.MODEL_DIR, 'final_model.pth'))
        self.logger.save_metrics()
        self.logger.plot_training_curves()
        
        # Statistiques finales
        final_stats = self.agent.get_training_stats()
        print(f"\n📊 Statistiques finales:")
        for key, value in final_stats.items():
            print(f"  {key}: {value:.4f}")
        
        return self.logger.metrics


def main():
    """Fonction principale d'entraînement"""
    print("🎯 Démarrage de l'entraînement de l'agent SAC pour la gestion de portefeuille")
    
    # Configuration personnalisée pour un entraînement plus rapide
    config_overrides = {
        'MAX_EPISODES': 200,  # Réduire pour le test
        'EVAL_FREQUENCY': 20,
        'SAVE_FREQUENCY': 50,
        'BATCH_SIZE': 64,  # Réduire la taille de batch
    }
    
    # Créer et lancer l'entraîneur
    trainer = PortfolioTrainer(config_overrides)
    
    try:
        metrics = trainer.train(num_episodes=config_overrides['MAX_EPISODES'])
        print("\n🎉 Entraînement réussi!")
        
        # Afficher quelques métriques finales
        if metrics['total_return']:
            final_return = metrics['total_return'][-1]
            max_return = max(metrics['total_return'])
            print(f"   Retour final: {final_return:.2%}")
            print(f"   Meilleur retour: {max_return:.2%}")
        
    except Exception as e:
        print(f"❌ Erreur durant l'entraînement: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
