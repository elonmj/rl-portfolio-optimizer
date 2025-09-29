"""
Script d'entraînement de l'agent SAC pour la gestion de portefeuille.
Implémente la boucle d'entraînement et validation selon spec.md.
Intègre le support Kaggle pour l'exécution GPU distante.
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
from typing import Dict, List, Tuple, Optional
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Kaggle compatibility: Add current directory to Python path
if '/kaggle' in os.getcwd():
    sys.path.insert(0, '/kaggle/src')
    sys.path.insert(0, '/kaggle/working')
    print(f"Kaggle environment detected. Working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:5]}")  # Show first 5 paths
    
    # List files in current directory for debugging
    try:
        import os
        print(f"Files in /kaggle/src: {os.listdir('/kaggle/src')}")
    except:
        print("Could not list /kaggle/src files")

try:
    from config import Config
    print("✅ Successfully imported Config")
except ImportError as e:
    print(f"❌ Failed to import Config: {e}")
    # Fallback for debugging
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    raise

from data_processing import DataHandler, FeatureProcessor
from environment import PortfolioEnv
from agent import SACAgent

# Kaggle integration (optional import)
try:
    from kaggle_manager import KaggleManager, KaggleConfig
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("Kaggle integration not available. Running in local mode only.")

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
        
    # train.py -> class TrainingLogger

    def log_episode(self, episode: int, metrics: Dict):
        """Enregistre les métriques d'un épisode de manière robuste"""
        self.metrics['episode'].append(episode)
        # Parcourir toutes les clés de métriques possibles
        for key in self.metrics:
            if key != 'episode':
                # Utiliser .get(key, None) pour ajouter la valeur si elle existe, ou None sinon.
                # Cela garantit que toutes les listes ont la même longueur.
                self.metrics[key].append(metrics.get(key, None))
        
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
    """Classe principale pour l'entraînement de l'agent SAC avec support Kaggle"""
    
    def __init__(self, config_overrides: Dict = None, kaggle_mode: bool = None):
        self.config = Config()
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(self.config, key, value)
        
        # Kaggle integration setup
        self.kaggle_mode = kaggle_mode if kaggle_mode is not None else Config.is_kaggle_environment()
        self.execution_mode = Config.get_execution_mode()
        self.kaggle_manager = None
        
        if self.kaggle_mode and KAGGLE_AVAILABLE:
            try:
                self.kaggle_manager = KaggleManager()
                print(f"🚀 Kaggle mode activated - Execution: {self.execution_mode}")
            except Exception as e:
                print(f"⚠️ Kaggle manager failed to initialize: {e}")
                self.kaggle_mode = False
        
        # Setup environment-specific paths
        Config.setup_environment_paths()
        self.paths = Config.get_data_paths()
        
        # Initialiser les composants
        self.data_handler = DataHandler()
        self.feature_processor = FeatureProcessor(self.data_handler)
        
        # Créer les répertoires nécessaires avec paths adaptés
        os.makedirs(self.paths['logs'], exist_ok=True)
        os.makedirs(self.paths['models'], exist_ok=True)
        os.makedirs(self.paths['results'], exist_ok=True)
        
        # Logger with environment-specific path
        self.logger = TrainingLogger(self.paths['logs'])
        
        # Kaggle-specific configuration
        if self.kaggle_mode:
            self.kaggle_config = Config.get_kaggle_config()
            self.training_config = Config.get_training_config_for_environment()
            print(f"📋 Kaggle training config: {self.training_config}")
        else:
            self.kaggle_config = None
            self.training_config = Config.get_training_config_for_environment()
        
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
        max_tickers = min(len(train_tickers), 6)  # Maximum 6 tickers
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
        """Configure l'agent SAC avec support Kaggle"""
        print("🤖 Configuration de l'agent SAC...")

        if self.train_env is None:
            raise ValueError("Environnement d'entraînement non configuré")

        # Dimensions
        num_assets = len(self.train_tickers)
        state_dim = self.train_env.observation_space.shape[0]
        action_dim = self.train_env.action_space.shape[0]

        # Device selection logic:
        # --kaggle mode: Force GPU if available (simulate Kaggle environment)
        # --local mode: Use standard device detection
        if self.kaggle_mode:
            # Kaggle workflow mode: prioritize GPU for Kaggle-like configuration
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"🚀 Kaggle workflow: GPU CUDA activé (simulation environnement Kaggle)")
            else:
                device = torch.device("cpu") 
                print(f"⚠️ Kaggle workflow: GPU non disponible, utilisation CPU")
        else:
            # Local mode: use standard device detection
            device = Config.init_device()
            print(f"🏠 Mode local: device standard détecté")
        
        print(f"🖥️ Device final: {device}")

        # Création de l'agent
        self.agent = SACAgent(
            num_assets=num_assets,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device
        )
        
        # Log Kaggle-specific configuration
        if self.kaggle_mode:
            print(f"🚀 Kaggle GPU enabled: {self.kaggle_config.get('enable_gpu', False)}")
            print(f"🌐 Kaggle Internet: {self.kaggle_config.get('enable_internet', False)}")
    
    def generate_kaggle_kernel_metadata(self, 
                                       task_type: str = "training",
                                       description: Optional[str] = None) -> Optional[str]:
        """
        Generate kernel metadata for Kaggle submission.
        
        Args:
            task_type: Type of task (training, evaluation, etc.)
            description: Custom description for the kernel
            
        Returns:
            Path to generated metadata file or None if not in Kaggle mode
        """
        if not self.kaggle_mode or not self.kaggle_manager:
            return None
            
        kernel_name = self.kaggle_manager.generate_kernel_name(task_type)
        title = f"RL Portfolio Optimizer - {task_type.title()}"
        
        if description is None:
            description = (
                f"SAC Agent Portfolio Optimization - {task_type} phase with GPU acceleration. "
                f"Using {len(self.train_tickers)} assets for portfolio management with "
                f"CVaR risk management and technical indicators."
            )
        
        # Custom configuration for this specific task
        custom_config = {
            "enable_gpu": self.kaggle_config.get('enable_gpu', True),
            "enable_internet": self.kaggle_config.get('enable_internet', True),
            "keywords": self.kaggle_config.get('keywords', []) + [task_type, "sac-agent"],
            "description": description
        }
        
        metadata = self.kaggle_manager.create_kernel_metadata(
            kernel_name=kernel_name,
            title=title,
            code_file="train.py",
            description=description,
            custom_config=custom_config
        )
        
        # Save metadata to working directory
        metadata_path = self.kaggle_manager.save_kernel_metadata(
            metadata, self.paths['working']
        )
        
        print(f"📄 Kernel metadata generated: {metadata_path}")
        return metadata_path
    
    def save_kaggle_results(self, final_stats: Dict, training_metrics: pd.DataFrame) -> None:
        """
        Save training results in Kaggle-compatible format.
        
        Args:
            final_stats: Final training statistics
            training_metrics: Training metrics DataFrame
        """
        if not self.kaggle_mode:
            return
            
        results_dir = self.paths['results']
        
        # Save comprehensive results JSON
        kaggle_results = {
            'execution_mode': self.execution_mode,
            'kaggle_config': self.kaggle_config,
            'training_config': self.training_config,
            'tickers_used': self.train_tickers,
            'final_statistics': final_stats,
            'training_summary': {
                'total_episodes': len(training_metrics),
                'best_reward': training_metrics['total_reward'].max() if not training_metrics.empty else 0,
                'final_portfolio_value': training_metrics['portfolio_value'].iloc[-1] if not training_metrics.empty else 0,
                'device_used': str(Config.get_device_for_kaggle()),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        results_file = os.path.join(results_dir, 'kaggle_training_results.json')
        with open(results_file, 'w') as f:
            json.dump(kaggle_results, f, indent=2, default=str)
            
        print(f"💾 Kaggle results saved to: {results_file}")
        
        # Also save metrics in standard CSV format for easy analysis
        metrics_file = os.path.join(results_dir, 'training_metrics_kaggle.csv')
        training_metrics.to_csv(metrics_file, index=False)
        print(f"📊 Training metrics saved to: {metrics_file}")

    
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
    
    def train(self, num_episodes: int = None):
        """Boucle principale d'entraînement avec support Kaggle"""
        # Use environment-specific episode count
        if num_episodes is None:
            num_episodes = self.training_config.get('max_episodes', Config.MAX_EPISODES)
            
        print(f"🏋️ Démarrage de l'entraînement pour {num_episodes} épisodes")
        print(f"🌍 Environnement: {self.execution_mode}")
        print(f"🚀 Workflow Kaggle: {'Oui' if self.kaggle_mode else 'Non'}")
        
        # Configuration
        self.setup_environments()
        self.setup_agent()
        
        # Generate Kaggle kernel metadata if in Kaggle mode (after environment setup)
        if self.kaggle_mode:
            self.generate_kaggle_kernel_metadata("training")
        
        # Log environment info
        Config.log_environment_info()
        
        start_time = time.time()
        
        for episode in tqdm(range(num_episodes), desc="Entraînement"):
            # Entraîner un épisode
            train_metrics = self.train_episode(episode)
            
            # Validation périodique
            val_metrics = {}
            eval_freq = self.training_config.get('eval_frequency', Config.EVAL_FREQUENCY)
            if episode % eval_freq == 0:
                val_metrics = self.validate_agent()
                
                # Sauvegarder le meilleur modèle avec path adapté
                val_return = val_metrics.get('val_total_return', -np.inf)
                if val_return > self.best_val_return:
                    self.best_val_return = val_return
                    best_model_path = os.path.join(self.paths['models'], 'best_model.pth')
                    self.agent.save(best_model_path)
                    if self.kaggle_mode:
                        print(f"💎 New best model saved: {best_model_path}")
            
            # Logger les métriques
            all_metrics = {**train_metrics, **val_metrics}
            self.logger.log_episode(episode, all_metrics)
            
            # Affichage périodique (plus fréquent en mode Kaggle pour monitoring)
            display_freq = 5 if self.kaggle_mode else 10
            if episode % display_freq == 0:
                train_return = train_metrics.get('total_return', 0)
                val_return = val_metrics.get('val_total_return', 0)
                print(f"\nÉpisode {episode}:")
                print(f"  Train Return: {train_return:.2%}")
                if val_metrics:
                    print(f"  Val Return: {val_return:.2%}")
                    print(f"  Val Sharpe: {val_metrics.get('val_sharpe_ratio', 0):.2f}")
                
                # Kaggle-specific monitoring
                if self.kaggle_mode:
                    portfolio_value = train_metrics.get('portfolio_value', Config.INITIAL_CASH)
                    print(f"  Portfolio Value: {portfolio_value:,.0f} FCFA")
                    print(f"  Device: {Config.get_device_for_kaggle()}")
            
            # Sauvegarde périodique avec path adapté
            save_freq = self.training_config.get('save_frequency', Config.SAVE_FREQUENCY)
            if episode % save_freq == 0 and episode > 0:
                model_path = os.path.join(self.paths['models'], f'model_episode_{episode}.pth')
                self.agent.save(model_path)
                self.logger.save_metrics()
                self.logger.plot_training_curves()
                
                if self.kaggle_mode:
                    print(f"💾 Checkpoint saved: episode {episode}")
        
        # Fin de l'entraînement
        total_time = time.time() - start_time
        print(f"\n✅ Entraînement terminé en {total_time/60:.1f} minutes")
        print(f"   Meilleur retour validation: {self.best_val_return:.2%}")
        
        # Sauvegarde finale avec path adapté
        final_model_path = os.path.join(self.paths['models'], 'final_model.pth')
        self.agent.save(final_model_path)
        self.logger.save_metrics()
        self.logger.plot_training_curves()
        
        # Statistiques finales
        final_stats = self.agent.get_training_stats()
        print(f"\n📊 Statistiques finales:")
        for key, value in final_stats.items():
            print(f"  {key}: {value:.4f}")
        
        # Kaggle-specific result saving
        if self.kaggle_mode:
            training_df = pd.DataFrame(self.logger.metrics)
            self.save_kaggle_results(final_stats, training_df)
            print(f"🚀 Kaggle training completed successfully!")
            print(f"📁 Results saved in: {self.paths['results']}")
        
        return self.logger.metrics


def main(num_episodes: Optional[int] = None, kaggle_mode: bool = False, github_mode: bool = False):
    """
    Fonction principale d'entraînement avec support Kaggle
    
    Args:
        num_episodes: Number of training episodes (None for config default)
        kaggle_mode: Active le workflow Kaggle API (upload/monitoring/download)
        github_mode: Active le mode GitHub pour tests
    """
    print("🎯 Démarrage de l'entraînement de l'agent SAC pour la gestion de portefeuille")
    
    # Real execution environment detection (where the code actually runs)
    execution_environment = Config.get_execution_mode()  # 'local' or 'kaggle'
    
    # Workflow mode (whether to use Kaggle API and automation)
    if kaggle_mode is None:
        kaggle_mode = False  # By default, use local workflow
    
    print(f"� Environnement d'exécution: {execution_environment}")
    print(f"🚀 Workflow Kaggle: {'Activé' if kaggle_mode else 'Désactivé'}")
    
    # Configuration selon le workflow choisi
    if kaggle_mode:
        # Configuration optimisée pour execution Kaggle (GPU + plus d'épisodes)
        config_overrides = {
            'MAX_EPISODES': num_episodes or 500,  # Plus d'épisodes avec GPU
            'EVAL_FREQUENCY': 25,
            'SAVE_FREQUENCY': 50,
            'BATCH_SIZE': 256,  # Batch size optimisé pour GPU
        }
        print("⚡ Configuration pour workflow Kaggle (GPU optimisé)")
    else:
        # Configuration pour workflow local standard
        config_overrides = {
            'MAX_EPISODES': num_episodes or 200,  # Test local plus rapide
            'EVAL_FREQUENCY': 20,
            'SAVE_FREQUENCY': 50,
            'BATCH_SIZE': 64,  # Batch size adapté au CPU local
        }
        print("🏠 Configuration pour workflow local standard")
    
    # Workflow Kaggle : upload + execution distante + monitoring + download
    if kaggle_mode:
        
        # Créer le manager Kaggle
        kaggle_manager = KaggleManager()
        
        if github_mode:
            print("🚀 Lancement du workflow Kaggle + GitHub...")
            
            try:
                # Nouveau workflow GitHub (plus fiable)
                results = kaggle_manager.create_and_upload_notebook_github(
                    repo_url="https://github.com/elonmj/rl-portfolio-optimizer.git",
                    branch="feature/training-config-updates",
                    task_type="training",
                    timeout=3600,  # 1 hour timeout
                    check_interval=30  # Check every 30s
                )
            except Exception as e:
                print(f"❌ Erreur du workflow GitHub: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        else:
            print("🚀 Lancement du workflow Dataset + Notebook...")
            
            # Préparer les fichiers à uploader
            source_files = [
                "train.py", "config.py", "agent.py", "models.py", 
                "environment.py", "data_processing.py", "utils.py"
            ]
            data_files = ["datas", "requirements.txt"]
            
            try:
                # Workflow Dataset + Notebook (ancien système)
                results = kaggle_manager.create_dataset_and_notebook_workflow(
                    source_files=source_files,
                    data_files=data_files,
                    task_type="training",
                    episodes=config_overrides['MAX_EPISODES']
                )
            except Exception as e:
                print(f"❌ Erreur du workflow Dataset + Notebook: {e}")
                import traceback
                traceback.print_exc()
                return None
            
            print(f"\n🎉 Workflow Dataset + Notebook terminé avec succès!")
            print(f"� Dataset: https://www.kaggle.com/datasets/{results['dataset_slug']}")
            print(f"📓 Notebook: https://www.kaggle.com/code/{results['kernel_slug']}")
            print(f"📊 Episodes: {results['episodes']}")
            print("\n📝 Instructions de monitoring:")
            print("1. Allez sur l'URL du notebook ci-dessus")
            print("2. Cliquez sur 'Run All' pour lancer l'entraînement") 
            print("3. Surveillez les logs en temps réel")
            print("4. Les résultats seront sauvegardés automatiquement")
            
            return results
    
    # Workflow local : execution standard sur la machine locale
    else:
        print("🏠 Lancement du workflow local...")
        
        # Créer et lancer l'entraîneur localement
        trainer = PortfolioTrainer(config_overrides, kaggle_mode=False)
        
        try:
            metrics = trainer.train(num_episodes=config_overrides['MAX_EPISODES'])
            print("\n🎉 Entraînement local réussi!")
            
            # Afficher quelques métriques finales
            if metrics['total_return']:
                final_return = metrics['total_return'][-1]
                max_return = max(metrics['total_return'])
                print(f"   Retour final: {final_return:.2%}")
                print(f"   Meilleur retour: {max_return:.2%}")
                
            final_value = metrics['portfolio_value'][-1] if metrics['portfolio_value'] else Config.INITIAL_CASH
            print(f"   Valeur finale du portefeuille: {final_value:,.0f} FCFA")
            print(f"   Device utilisé: {Config.get_device()}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Erreur durant l'entraînement local: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entraînement RL Portfolio Optimizer")
    parser.add_argument('--kaggle', action='store_true', help='Force Kaggle mode (dataset-based)')
    parser.add_argument('--kaggle-github', action='store_true', help='Kaggle mode with GitHub cloning')
    parser.add_argument('--local', action='store_true', help='Force local mode')
    parser.add_argument('--episodes', type=int, help='Number of training episodes')
    
    args = parser.parse_args()
    
    # Determine workflow mode
    if args.kaggle:
        kaggle_mode = True
        github_mode = False
    elif args.kaggle_github:
        kaggle_mode = True  
        github_mode = True
    elif args.local:
        kaggle_mode = False
        github_mode = False
    else:
        kaggle_mode = False  # Default to local workflow
        github_mode = False
    
    main(kaggle_mode=kaggle_mode, github_mode=github_mode, num_episodes=args.episodes)
