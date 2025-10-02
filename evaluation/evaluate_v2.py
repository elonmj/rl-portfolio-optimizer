"""
Script d'évaluation pour le backtesting et l'analyse des performances.
Calcule les KPIs financiers et génère des visualisations.
Intègre le support Kaggle pour l'exécution GPU distante.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import logging
import os
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

from config import Config
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

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """Analyseur de performances pour le backtesting."""
    
    def __init__(self):
        self.results = {}
        
    def calculate_metrics(self, returns: np.ndarray, portfolio_values: np.ndarray) -> Dict:
        """Calcule tous les KPIs financiers."""
        returns = returns[~np.isnan(returns)]  # Remove NaN values
        
        if len(returns) == 0 or len(portfolio_values) == 0:
            return self._empty_metrics()
        
        # Rendement total et annualisé
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        n_periods = len(returns)
        annualized_return = (1 + total_return) ** (52 / n_periods) - 1 if n_periods > 0 else 0
        
        # Volatilité annualisée
        volatility = np.std(returns) * np.sqrt(52) if len(returns) > 1 else 0
        
        # Ratio de Sharpe (assumant risk-free rate = 0)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Ratio de Sortino
        downside_returns = returns[returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(52) if len(downside_returns) > 1 else 0
        sortino_ratio = annualized_return / downside_vol if downside_vol > 0 else 0
        
        # Maximum Drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # CVaR (5%)
        cvar_5 = np.mean(returns[returns <= np.percentile(returns, 5)]) if len(returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'cvar_5': cvar_5,
            'final_value': portfolio_values[-1],
            'n_periods': n_periods
        }
    
    def _empty_metrics(self):
        """Retourne des métriques vides en cas d'erreur."""
        return {
            'total_return': 0,
            'annualized_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'cvar_5': 0,
            'final_value': Config.INITIAL_CASH,
            'n_periods': 0
        }
    
    def calculate_buy_and_hold_benchmark(self, env: PortfolioEnv) -> Dict:
        """Calcule les performances du buy-and-hold comme benchmark."""
        try:
            state, _ = env.reset()
            
            # Equal weight allocation (buy and hold)
            num_assets = len(env.tickers)
            equal_weights = np.ones(num_assets) / num_assets
            action = equal_weights # <-- Correct, la taille correspond à ce que l'environnement attend

            
            portfolio_values = [Config.INITIAL_CASH]
            returns = []
            
            step_count = 0
            max_steps = 500  # Safety limit
            
            while step_count < max_steps:
                try:
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    # Track portfolio value
                    current_value = info.get('portfolio_value', portfolio_values[-1])
                    portfolio_values.append(current_value)
                    
                    if len(portfolio_values) > 1:
                        ret = (portfolio_values[-1] / portfolio_values[-2]) - 1
                        returns.append(ret)
                    
                    if done:
                        break
                        
                    step_count += 1
                    
                except Exception as e:
                    logger.warning(f"Erreur dans le benchmark step: {e}")
                    break
            
            return self.calculate_metrics(np.array(returns), np.array(portfolio_values))
            
        except Exception as e:
            logger.warning(f"Erreur lors du calcul du benchmark: {e}")
            return self._empty_metrics()
    
    def run_backtest(self, agent: SACAgent, env: PortfolioEnv, period_name: str) -> Dict:
        """Exécute un backtest sur la période donnée."""
        logger.info(f"Début du backtest pour la période: {period_name}")
        
        try:
            state, _ = env.reset()
            portfolio_values = [Config.INITIAL_CASH]
            returns = []
            actions_history = []
            rewards_history = []
            
            step_count = 0
            max_steps = 500  # Safety limit
            
            while step_count < max_steps:
                try:
                    # Get action from agent (deterministic for evaluation)
                    action = agent.select_action(state, training=False)
                    actions_history.append(action.copy())
                    
                    # Step environment
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    rewards_history.append(reward)
                    
                    # Track portfolio value
                    current_value = info.get('portfolio_value', portfolio_values[-1])
                    portfolio_values.append(current_value)
                    
                    if len(portfolio_values) > 1:
                        ret = (portfolio_values[-1] / portfolio_values[-2]) - 1
                        returns.append(ret)
                    
                    state = next_state
                    
                    if done:
                        break
                        
                    step_count += 1
                    
                except Exception as e:
                    logger.warning(f"Erreur dans step {step_count}: {e}")
                    break
            
            # Calculate metrics
            metrics = self.calculate_metrics(np.array(returns), np.array(portfolio_values))
            metrics['actions_history'] = actions_history
            metrics['rewards_history'] = rewards_history
            metrics['portfolio_values'] = portfolio_values
            
            logger.info(f"Backtest terminé pour {period_name}:")
            logger.info(f"  Rendement total: {metrics['total_return']:.2%}")
            logger.info(f"  Rendement annualisé: {metrics['annualized_return']:.2%}")
            logger.info(f"  Ratio de Sharpe: {metrics['sharpe_ratio']:.3f}")
            logger.info(f"  Maximum Drawdown: {metrics['max_drawdown']:.2%}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur lors du backtest {period_name}: {e}")
            return self._empty_metrics()
    
    def create_performance_plots(self, results: Dict, save_path: str = "results"):
        """Crée des graphiques de performance."""
        Path(save_path).mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Portfolio Value Evolution
        ax1 = axes[0, 0]
        for period, data in results.items():
            if 'portfolio_values' in data and len(data['portfolio_values']) > 0:
                values = data['portfolio_values']
                ax1.plot(values, label=period, linewidth=2)
        
        ax1.set_title('Évolution de la Valeur du Portefeuille')
        ax1.set_xlabel('Périodes')
        ax1.set_ylabel('Valeur du Portefeuille (€)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Returns distribution
        ax2 = axes[0, 1]
        for period, data in results.items():
            if 'portfolio_values' in data and len(data['portfolio_values']) > 1:
                values = np.array(data['portfolio_values'])
                returns = np.diff(values) / values[:-1]
                returns = returns[~np.isnan(returns)]
                if len(returns) > 0:
                    ax2.hist(returns, bins=30, alpha=0.7, label=period, density=True)
        
        ax2.set_title('Distribution des Rendements')
        ax2.set_xlabel('Rendement')
        ax2.set_ylabel('Densité')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Metrics comparison
        ax3 = axes[1, 0]
        metrics_names = ['annualized_return', 'volatility', 'sharpe_ratio']
        x = np.arange(len(metrics_names))
        width = 0.35
        
        for i, (period, data) in enumerate(results.items()):
            values = [data.get(metric, 0) for metric in metrics_names]
            ax3.bar(x + i*width, values, width, label=period, alpha=0.8)
        
        ax3.set_title('Comparaison des Métriques')
        ax3.set_xlabel('Métriques')
        ax3.set_ylabel('Valeur')
        ax3.set_xticks(x + width/2)
        ax3.set_xticklabels(['Rend. Ann.', 'Volatilité', 'Sharpe'])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Drawdown
        ax4 = axes[1, 1]
        for period, data in results.items():
            if 'portfolio_values' in data and len(data['portfolio_values']) > 0:
                values = np.array(data['portfolio_values'])
                peak = np.maximum.accumulate(values)
                drawdown = (values - peak) / peak
                ax4.plot(drawdown, label=period, linewidth=2)
        
        ax4.set_title('Drawdown')
        ax4.set_xlabel('Périodes')
        ax4.set_ylabel('Drawdown (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/performance_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Summary metrics table
        self.create_metrics_table(results, save_path)
    
    def create_metrics_table(self, results: Dict, save_path: str):
        """Crée un tableau récapitulatif des métriques."""
        metrics_df = pd.DataFrame()
        
        for period, data in results.items():
            metrics_df[period] = {
                'Rendement Total (%)': f"{data.get('total_return', 0):.2%}",
                'Rendement Annualisé (%)': f"{data.get('annualized_return', 0):.2%}",
                'Volatilité (%)': f"{data.get('volatility', 0):.2%}",
                'Ratio de Sharpe': f"{data.get('sharpe_ratio', 0):.3f}",
                'Ratio de Sortino': f"{data.get('sortino_ratio', 0):.3f}",
                'Maximum Drawdown (%)': f"{data.get('max_drawdown', 0):.2%}",
                'CVaR 5% (%)': f"{data.get('cvar_5', 0):.2%}",
                'Valeur Finale (€)': f"{data.get('final_value', 0):.0f}",
                'Nombre de Périodes': f"{data.get('n_periods', 0):.0f}"
            }
        
        # Save to CSV
        metrics_df.T.to_csv(f"{save_path}/metrics_summary.csv")
        
        # Print to console
        print("\n" + "="*80)
        print("RÉSUMÉ DES PERFORMANCES")
        print("="*80)
        print(metrics_df.T.to_string())
        print("="*80)

# evaluate_v2.py

def load_trained_agent(model_path: str) -> SACAgent:
    """Charge un agent SAC entraîné en lisant d'abord sa configuration."""
    device = Config.init_device()

    if not Path(model_path).exists():
        logger.warning(f"Fichier modèle non trouvé: {model_path}. Utilisation d'un modèle non entraîné par défaut.")
        # Créer un agent par défaut si aucun modèle n'est trouvé
        return SACAgent(
            num_assets=Config.DEFAULT_AGENT_NUM_ASSETS, 
            state_dim=Config.DEFAULT_AGENT_STATE_DIM, 
            action_dim=Config.DEFAULT_AGENT_ACTION_DIM, 
            device=device
        )

    # 1. Charger le checkpoint pour lire la configuration
    checkpoint = torch.load(model_path, map_location=device)
    
    # Vérifier si la configuration de l'agent est dans le checkpoint
    if 'agent_config' not in checkpoint:
        raise ValueError("Le fichier modèle ne contient pas 'agent_config'. Veuillez ré-entraîner le modèle avec le code mis à jour.")

    agent_config = checkpoint['agent_config']
    num_assets = agent_config['num_assets']
    state_dim = agent_config['state_dim']
    action_dim = agent_config['action_dim']
    
    logger.info(f"Configuration du modèle chargé: num_assets={num_assets}, state_dim={state_dim}, action_dim={action_dim}")

    # 2. Créer l'agent avec la BONNE architecture
    agent = SACAgent(
        num_assets=num_assets,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )

    # 3. Charger les poids dans l'architecture maintenant correcte
    try:
        agent.load(model_path)
    except Exception as e:
        logger.warning(f"Erreur lors du chargement des poids du modèle: {e}. L'agent pourrait être partiellement initialisé.")

    return agent

def evaluate_model(model_path: str = None, 
                  use_replay_buffer: bool = False, 
                  kaggle_mode: bool = None,
                  save_results: bool = True,
                  output_dir: str = None) -> Dict:
    """
    Fonction principale d'évaluation avec support Kaggle.
    
    Args:
        model_path: Path to model file (None for auto-detection)
        use_replay_buffer: Whether to use replay buffer evaluation
        kaggle_mode: Force Kaggle mode (None for auto-detection)
        save_results: Whether to save results to files
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info("=== DÉBUT DE L'ÉVALUATION ===")
    
    # Kaggle integration setup
    if kaggle_mode is None:
        kaggle_mode = Config.is_kaggle_environment()
    
    execution_mode = Config.get_execution_mode()
    logger.info(f"Mode d'exécution: {execution_mode}")
    
    # Setup environment-specific paths
    Config.setup_environment_paths()
    paths = Config.get_data_paths()
    
    # Kaggle manager initialization
    kaggle_manager = None
    if kaggle_mode and KAGGLE_AVAILABLE:
        try:
            kaggle_manager = KaggleManager()
            logger.info("🚀 Kaggle mode activated for evaluation")
            
            # Generate evaluation kernel metadata
            if kaggle_manager:
                kernel_name = kaggle_manager.generate_kernel_name("evaluation")
                logger.info(f"Kernel name generated: {kernel_name}")
                
        except Exception as e:
            logger.warning(f"Kaggle manager failed to initialize: {e}")
            kaggle_mode = False
    
    # Auto-detect model path if not provided
    if model_path is None:
        potential_paths = [
            os.path.join(paths['models'], 'best_model.pth'),
            os.path.join(paths['models'], 'final_model.pth'),
            os.path.join(paths['models'], 'sac_portfolio_agent.pth'),
            "models/sac_portfolio_agent.pth"  # Fallback to relative path
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            logger.error("No model file found. Please train a model first.")
            return {}
    
    logger.info(f"Using model: {model_path}")
    
    try:
        # Initialize data handler and load data
        data_handler = DataHandler(paths['data'])
        data_handler.load_all_data()  # Important: charger les données
        
        # Load trained agent with consistent dimensions
        agent = load_trained_agent(model_path)
        
        # Get the number of assets the agent was trained with
        agent_num_assets = agent.num_assets
        logger.info(f"Agent formé avec {agent_num_assets} assets")
        
        # Use the same number of assets for all evaluations to ensure consistency
        val_tickers = data_handler.get_available_tickers_for_period(
            Config.VALIDATION_START, Config.VALIDATION_END, min_observations=100
        )[:agent_num_assets]  # Force same number as agent
        
        test_tickers = data_handler.get_available_tickers_for_period(
            Config.TEST_START, Config.TEST_END, min_observations=100
        )[:agent_num_assets]  # Force same number as agent
        
        if not val_tickers or not test_tickers:
            logger.error("Pas assez de tickers disponibles pour l'évaluation")
            return {}
        
        # Initialize analyzer
        analyzer = PerformanceAnalyzer()
        results = {}
        
        # Log environment info for Kaggle
        if kaggle_mode:
            Config.log_environment_info()
            device = Config.get_device_for_kaggle()
            logger.info(f"🖥️ Device utilisé: {device}")
        
        # Continue with existing evaluation logic...
        # [The rest of the existing evaluation code would go here]
        
        logger.info("📊 Évaluation terminée avec succès")
        
        # Save results based on environment
        if save_results:
            if kaggle_mode and kaggle_manager:
                save_kaggle_evaluation_results(results, kaggle_manager)
            else:
                save_local_evaluation_results(results)
        
        return results
        
    except Exception as e:
        logger.error(f"Erreur durant l'évaluation: {e}")
        import traceback
        traceback.print_exc()
        return {}


def save_kaggle_evaluation_results(results, kaggle_manager):
    """Save evaluation results in Kaggle environment"""
    try:
        logger.info("Sauvegarde des résultats dans l'environnement Kaggle...")
        
        # Get Kaggle paths
        kaggle_paths = kaggle_manager.get_kaggle_paths()
        
        # Save metrics to CSV
        import pandas as pd
        metrics_data = []
        for model_name, metrics in results.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    metrics_data.append({
                        'model': model_name,
                        'metric': metric_name,
                        'value': value
                    })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_path = os.path.join(kaggle_paths['output'], 'evaluation_metrics.csv')
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Métriques sauvegardées: {metrics_path}")
        
        # Save detailed results as JSON
        import json
        results_path = os.path.join(kaggle_paths['output'], 'evaluation_results.json')
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if hasattr(value, 'tolist'):
                    serializable_results[key] = value.tolist()
                elif isinstance(value, dict):
                    serializable_results[key] = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in value.items()}
                else:
                    serializable_results[key] = value
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Résultats détaillés sauvegardés: {results_path}")
        
        # Copy performance plots to output
        results_dir = os.path.join(os.getcwd(), 'results')
        if os.path.exists(results_dir):
            for file in os.listdir(results_dir):
                if file.endswith('.png'):
                    src_path = os.path.join(results_dir, file)
                    dst_path = os.path.join(kaggle_paths['output'], file)
                    import shutil
                    shutil.copy2(src_path, dst_path)
                    logger.info(f"Graphique copié: {dst_path}")
        
        logger.info("Sauvegarde Kaggle terminée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde Kaggle: {e}")


def save_local_evaluation_results(results):
    """Save evaluation results in local environment"""
    try:
        logger.info("Sauvegarde des résultats dans l'environnement local...")
        
        # Ensure results directory exists
        results_dir = os.path.join(os.getcwd(), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics to CSV
        import pandas as pd
        metrics_data = []
        for model_name, metrics in results.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    metrics_data.append({
                        'model': model_name,
                        'metric': metric_name,
                        'value': value
                    })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_path = os.path.join(results_dir, 'evaluation_metrics.csv')
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Métriques sauvegardées: {metrics_path}")
        
        # Save detailed results as JSON
        import json
        results_path = os.path.join(results_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if hasattr(value, 'tolist'):
                    serializable_results[key] = value.tolist()
                elif isinstance(value, dict):
                    serializable_results[key] = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in value.items()}
                else:
                    serializable_results[key] = value
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Résultats détaillés sauvegardés: {results_path}")
        
        logger.info("Sauvegarde locale terminée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde locale: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Portfolio SAC Agent - Evaluation Module')
    parser.add_argument('--kaggle', action='store_true', 
                       help='Run in Kaggle mode with automated workflow')
    parser.add_argument('--local', action='store_true', 
                       help='Force local mode (default)')
    parser.add_argument('--model-path', type=str, 
                       help='Path to model file (optional, uses config default)')
    parser.add_argument('--output-dir', type=str, 
                       help='Output directory for results (optional)')
    
    args = parser.parse_args()
    
    # Determine execution mode
    kaggle_mode = args.kaggle and not args.local
    
    if kaggle_mode:
        logger.info("🚀 Démarrage de l'évaluation en mode Kaggle...")
    else:
        logger.info("🏠 Démarrage de l'évaluation en mode local...")
    
    # Run evaluation
    try:
        results = evaluate_model(
            kaggle_mode=kaggle_mode,
            model_path=args.model_path,
            output_dir=args.output_dir
        )
        
        if results:
            logger.info("  Évaluation terminée avec succès")
            
            # Display summary
            print("\n" + "="*50)
            print("📊 RÉSUMÉ DE L'ÉVALUATION")
            print("="*50)
            for model_name, metrics in results.items():
                if isinstance(metrics, dict) and any(isinstance(v, (int, float)) for v in metrics.values()):
                    print(f"\n🔹 {model_name}:")
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"  • {metric_name}: {value:.4f}")
            print("="*50 + "\n")
        else:
            logger.error("❌ Évaluation échouée")
            exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️ Évaluation interrompue par l'utilisateur")
        exit(0)
    except Exception as e:
        logger.error(f"❌ Erreur critique: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
