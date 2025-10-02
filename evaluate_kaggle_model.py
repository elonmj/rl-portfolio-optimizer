#!/usr/bin/env python3
"""
Évaluation complète du modèle Portfolio RL formé sur Kaggle
Conforme aux spécifications de modelisation.pdf
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Import des modules locaux
from environment import PortfolioEnv
from agent import SACAgent
from models import create_sac_models
from config import Config


def main():
    """
    Fonction principale d'évaluation du modèle Kaggle
    """
    print("🎯 ÉVALUATION DU MODÈLE PORTFOLIO RL FORMÉ SUR KAGGLE")
    print("=" * 60)
    
    # Vérifier que le modèle existe
    model_path = "models/sac_portfolio_agent_kaggle.pth"
    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé: {model_path}")
        return
    
    try:
        # Chargement des données
        print("\n📁 Chargement des données d'actions...")
        actions_data = pd.read_excel('datas/actions_secteurs_pays.xlsx')
        available_tickers = actions_data['Ticker'].tolist()
        
        # Sélectionner les tickers pour l'évaluation (6 premiers pour test)
        test_tickers = available_tickers[:6]
        
        print(f"   ✅ {len(available_tickers)} tickers disponibles")
        print(f"   🎯 Évaluation sur: {test_tickers}")
        
        # Créer l'environnement d'évaluation
        print(f"\n🔧 Configuration de l'environnement d'évaluation...")
        
        env = PortfolioEnv(
            tickers=test_tickers,
            start_date='2019-01-01',
            end_date='2019-12-31'
        )
        
        # Obtenir les dimensions
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        print(f"   - Observation space: {obs_dim} dimensions")
        print(f"   - Action space: {action_dim} dimensions") 
        print(f"   - Assets valides: {env.valid_tickers}")
        print(f"   - Périodes: {len(env.data)} pas de temps")
        
        # Créer et charger le modèle
        print(f"\n🤖 Chargement du modèle SAC Kaggle...")
        
        # Créer l'agent
        agent = SACAgent(
            state_dim=obs_dim,
            action_dim=action_dim,
            lr_actor=Config.SAC_LR_ACTOR,
            lr_critic=Config.SAC_LR_CRITIC,
            tau=Config.SAC_TAU,
            gamma=Config.SAC_GAMMA,
            alpha=Config.SAC_ALPHA
        )
        
        # Charger les poids du modèle Kaggle
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                if 'actor_state_dict' in checkpoint:
                    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
                    agent.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
                    agent.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
                    print(f"   ✅ Modèle chargé avec succès")
                else:
                    print(f"   ⚠️ Format de checkpoint non reconnu")
            else:
                print(f"   ⚠️ Utilisation du modèle initialisé")
                
        except Exception as e:
            print(f"   ⚠️ Erreur de chargement: {e}")
            print(f"   🔄 Utilisation du modèle initialisé")
        
        # Mettre en mode évaluation
        agent.actor.eval()
        agent.critic_1.eval() 
        agent.critic_2.eval()
        
        # Évaluation sur plusieurs épisodes
        print(f"\n📊 Début de l'évaluation (3 épisodes)...")
        
        os.makedirs('results', exist_ok=True)
        all_results = []
        
        for episode in range(3):
            print(f"\n🎮 Épisode {episode + 1}/3")
            
            # Reset de l'environnement
            obs = env.reset()
            terminated = False
            truncated = False
            step = 0
            
            episode_data = {
                'episode': episode,
                'portfolio_values': [],
                'rewards': [],
                'actions': [],
                'allocations': [],
                'transaction_costs': []
            }
            
            initial_value = env._calculate_portfolio_value()
            episode_data['portfolio_values'].append(initial_value)
            
            while not (terminated or truncated):
                # Action de l'agent
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action = agent.select_action(obs_tensor, evaluate=True)
                
                # Exécuter l'action
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Collecter les données
                episode_data['rewards'].append(reward)
                episode_data['actions'].append(action.copy())
                episode_data['allocations'].append(env.current_weights.copy())
                
                portfolio_value = env._calculate_portfolio_value()
                episode_data['portfolio_values'].append(portfolio_value)
                
                if hasattr(env, 'transaction_costs_history') and env.transaction_costs_history:
                    episode_data['transaction_costs'].append(env.transaction_costs_history[-1])
                else:
                    episode_data['transaction_costs'].append(0.0)
                
                obs = next_obs
                step += 1
                
                if step % 20 == 0:
                    print(f"   Step {step}: Value={portfolio_value:8.2f}, Reward={reward:7.4f}")
            
            # Calculer les métriques de l'épisode
            values = episode_data['portfolio_values']
            returns = [(values[i]/values[i-1] - 1) for i in range(1, len(values))]
            
            total_return = (values[-1] / values[0]) - 1 if len(values) > 1 else 0
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) if returns else 0
            
            # Drawdown
            peak = values[0]
            max_dd = 0
            for value in values[1:]:
                if value > peak:
                    peak = value
                else:
                    dd = (peak - value) / peak
                    max_dd = max(max_dd, dd)
            
            episode_metrics = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio, 
                'max_drawdown': max_dd,
                'mean_reward': np.mean(episode_data['rewards']),
                'total_transaction_costs': np.sum(episode_data['transaction_costs'])
            }
            
            episode_data['metrics'] = episode_metrics
            all_results.append(episode_data)
            
            print(f"   ✅ Épisode {episode + 1} terminé:")
            print(f"      - Steps: {step}")
            print(f"      - Rendement: {total_return:.2%}")
            print(f"      - Sharpe: {sharpe_ratio:.3f}")
            print(f"      - Max DD: {max_dd:.2%}")
        
        # Générer les graphiques
        print(f"\n📈 Génération des visualisations...")
        generate_evaluation_plots(all_results, env)
        
        # Générer le rapport
        print(f"\n📋 Génération du rapport...")
        generate_evaluation_report(all_results, env)
        
        # Résumé final
        avg_return = np.mean([ep['metrics']['total_return'] for ep in all_results])
        avg_sharpe = np.mean([ep['metrics']['sharpe_ratio'] for ep in all_results])
        avg_dd = np.mean([ep['metrics']['max_drawdown'] for ep in all_results])
        
        print(f"\n" + "="*60)
        print(f"🎯 RÉSUMÉ DE L'ÉVALUATION - MODÈLE KAGGLE")
        print("="*60)
        print(f"📊 3 épisodes évalués")
        print(f"📈 Rendement moyen: {avg_return:.2%}")
        print(f"⚡ Sharpe moyen: {avg_sharpe:.3f}")
        print(f"📉 Drawdown moyen: {avg_dd:.2%}")
        print("="*60)
        print("✅ Évaluation terminée avec succès!")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'évaluation: {e}")
        import traceback
        traceback.print_exc()


def generate_evaluation_plots(all_results, env):
    """
    Génère les graphiques d'évaluation
    """
    # 1. Performance des portfolios
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Valeurs du portfolio
    for i, ep in enumerate(all_results):
        values = ep['portfolio_values']
        steps = range(len(values))
        ax1.plot(steps, values, alpha=0.7, label=f'Épisode {i+1}')
    
    ax1.set_title('Évolution de la valeur du portfolio', fontweight='bold')
    ax1.set_xlabel('Pas de temps')
    ax1.set_ylabel('Valeur du portfolio')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rendements cumulés
    for i, ep in enumerate(all_results):
        values = ep['portfolio_values']
        if len(values) > 1:
            cum_returns = [(v / values[0] - 1) * 100 for v in values]
            steps = range(len(cum_returns))
            ax2.plot(steps, cum_returns, alpha=0.7, label=f'Épisode {i+1}')
    
    ax2.set_title('Rendements cumulés (%)', fontweight='bold')
    ax2.set_xlabel('Pas de temps')
    ax2.set_ylabel('Rendement cumulé (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Rewards
    for i, ep in enumerate(all_results):
        rewards = ep['rewards']
        steps = range(len(rewards))
        ax3.plot(steps, rewards, alpha=0.7, label=f'Épisode {i+1}')
    
    ax3.set_title('Évolution des rewards', fontweight='bold')
    ax3.set_xlabel('Pas de temps')
    ax3.set_ylabel('Reward')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Métriques par épisode
    episodes = range(1, len(all_results) + 1)
    returns = [ep['metrics']['total_return'] * 100 for ep in all_results]
    sharpes = [ep['metrics']['sharpe_ratio'] for ep in all_results]
    
    ax4_twin = ax4.twinx()
    
    bars1 = ax4.bar([x - 0.2 for x in episodes], returns, 
                   width=0.4, alpha=0.7, label='Rendement (%)', color='green')
    bars2 = ax4_twin.bar([x + 0.2 for x in episodes], sharpes, 
                       width=0.4, alpha=0.7, label='Sharpe', color='blue')
    
    ax4.set_title('Métriques par épisode', fontweight='bold')
    ax4.set_xlabel('Épisode')
    ax4.set_ylabel('Rendement (%)', color='green')
    ax4_twin.set_ylabel('Sharpe ratio', color='blue')
    ax4.set_xticks(episodes)
    
    plt.tight_layout()
    plt.savefig('results/evaluation_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Allocations
    n_assets = len(env.valid_tickers)
    colors = plt.cm.Set3(np.linspace(0, 1, n_assets))
    
    fig, axes = plt.subplots(len(all_results), 1, figsize=(12, 4*len(all_results)))
    if len(all_results) == 1:
        axes = [axes]
    
    for ep_idx, ep in enumerate(all_results):
        allocations = np.array(ep['allocations'])
        steps = range(len(allocations))
        
        axes[ep_idx].stackplot(steps, *allocations.T, labels=env.valid_tickers, 
                             colors=colors, alpha=0.8)
        axes[ep_idx].set_title(f'Allocations - Épisode {ep_idx + 1}', fontweight='bold')
        axes[ep_idx].set_xlabel('Pas de temps')
        axes[ep_idx].set_ylabel('Allocation')
        axes[ep_idx].set_ylim(0, 1)
        axes[ep_idx].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axes[ep_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/evaluation_allocations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Graphiques sauvegardés dans 'results/'")


def generate_evaluation_report(all_results, env):
    """
    Génère le rapport d'évaluation
    """
    report = []
    report.append("# 🎯 RAPPORT D'ÉVALUATION - MODÈLE PORTFOLIO RL KAGGLE")
    report.append("=" * 60)
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Configuration
    report.append("## 🔧 Configuration")
    report.append(f"- Assets: {', '.join(env.valid_tickers)}")
    report.append(f"- Nombre d'assets: {len(env.valid_tickers)}")
    report.append(f"- Périodes: {len(env.data)} pas de temps")
    report.append(f"- Dimensions état: {env.observation_space.shape[0]}")
    report.append("")
    
    # Métriques agrégées
    avg_return = np.mean([ep['metrics']['total_return'] for ep in all_results])
    avg_sharpe = np.mean([ep['metrics']['sharpe_ratio'] for ep in all_results])
    avg_dd = np.mean([ep['metrics']['max_drawdown'] for ep in all_results])
    avg_reward = np.mean([ep['metrics']['mean_reward'] for ep in all_results])
    
    report.append("## 📊 Métriques agrégées")
    report.append(f"- Épisodes: {len(all_results)}")
    report.append(f"- Rendement moyen: {avg_return:.2%}")
    report.append(f"- Sharpe moyen: {avg_sharpe:.3f}")
    report.append(f"- Drawdown moyen: {avg_dd:.2%}")
    report.append(f"- Reward moyen: {avg_reward:.4f}")
    report.append("")
    
    # Détails par épisode
    report.append("## 📈 Détails par épisode")
    for i, ep in enumerate(all_results):
        metrics = ep['metrics']
        report.append(f"### Épisode {i + 1}")
        report.append(f"- Rendement: {metrics['total_return']:.2%}")
        report.append(f"- Sharpe: {metrics['sharpe_ratio']:.3f}")
        report.append(f"- Max DD: {metrics['max_drawdown']:.2%}")
        report.append(f"- Reward moyen: {metrics['mean_reward']:.4f}")
        report.append(f"- Coûts: {metrics['total_transaction_costs']:.2f}")
        report.append("")
    
    # Framework mathématique
    report.append("## 🧮 Framework mathématique implémenté")
    report.append("- ✅ Espace d'état amélioré (7 composants)")
    report.append("- ✅ Récompense multi-composants")
    report.append("- ✅ Modélisation stochastique des risques")
    report.append("- ✅ Coûts de transaction")
    report.append("- ✅ Sélection d'actifs")
    report.append("")
    
    with open('results/evaluation_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("   ✅ Rapport sauvegardé: results/evaluation_report.md")


if __name__ == "__main__":
    main()