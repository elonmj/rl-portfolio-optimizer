"""
Utilitaires pour l'analyse et la visualisation des performances.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Optional

def plot_training_curves(log_file: str = "training_log.json", save_path: str = "results"):
    """Trace les courbes d'entraînement à partir du fichier de log."""
    Path(save_path).mkdir(exist_ok=True)
    
    # Load training data
    if not Path(log_file).exists():
        print(f"Fichier de log non trouvé: {log_file}")
        return
    
    with open(log_file, 'r') as f:
        training_data = [json.loads(line) for line in f]
    
    # Convert to DataFrame
    df = pd.DataFrame(training_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    plt.style.use('seaborn-v0_8')
    
    # 1. Portfolio Value
    ax1 = axes[0, 0]
    if 'portfolio_value' in df.columns:
        ax1.plot(df['episode'], df['portfolio_value'], linewidth=2, color='blue')
        ax1.set_title('Valeur du Portefeuille')
        ax1.set_xlabel('Épisode')
        ax1.set_ylabel('Valeur (€)')
        ax1.grid(True, alpha=0.3)
    
    # 2. Total Reward
    ax2 = axes[0, 1]
    if 'total_reward' in df.columns:
        # Smooth the reward curve
        window = min(50, len(df) // 10)
        if window > 1:
            smoothed_reward = df['total_reward'].rolling(window=window, center=True).mean()
            ax2.plot(df['episode'], df['total_reward'], alpha=0.3, color='gray', label='Raw')
            ax2.plot(df['episode'], smoothed_reward, linewidth=2, color='red', label=f'Smooth ({window})')
            ax2.legend()
        else:
            ax2.plot(df['episode'], df['total_reward'], linewidth=2, color='red')
        ax2.set_title('Récompense Totale')
        ax2.set_xlabel('Épisode')
        ax2.set_ylabel('Récompense')
        ax2.grid(True, alpha=0.3)
    
    # 3. Actor Loss
    ax3 = axes[0, 2]
    if 'actor_loss' in df.columns:
        valid_losses = df['actor_loss'].dropna()
        if len(valid_losses) > 0:
            episodes = df.loc[valid_losses.index, 'episode']
            ax3.plot(episodes, valid_losses, linewidth=2, color='green')
            ax3.set_title('Perte de l\'Acteur')
            ax3.set_xlabel('Épisode')
            ax3.set_ylabel('Perte')
            ax3.grid(True, alpha=0.3)
    
    # 4. Critic Loss
    ax4 = axes[1, 0]
    if 'critic_loss' in df.columns:
        valid_losses = df['critic_loss'].dropna()
        if len(valid_losses) > 0:
            episodes = df.loc[valid_losses.index, 'episode']
            ax4.plot(episodes, valid_losses, linewidth=2, color='orange')
            ax4.set_title('Perte du Critique')
            ax4.set_xlabel('Épisode')
            ax4.set_ylabel('Perte')
            ax4.grid(True, alpha=0.3)
    
    # 5. CVaR Evolution
    ax5 = axes[1, 1]
    if 'cvar' in df.columns:
        ax5.plot(df['episode'], df['cvar'], linewidth=2, color='purple')
        ax5.set_title('CVaR (5%)')
        ax5.set_xlabel('Épisode')
        ax5.set_ylabel('CVaR')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 6. Actions Distribution (last episode)
    ax6 = axes[1, 2]
    if 'final_weights' in df.columns and len(df) > 0:
        # Get the last episode's weights
        last_weights = df['final_weights'].iloc[-1]
        if isinstance(last_weights, (list, np.ndarray)) and len(last_weights) > 1:
            weights = np.array(last_weights[:-1])  # Exclude cash
            ax6.bar(range(len(weights)), weights, alpha=0.7, color='skyblue')
            ax6.set_title('Répartition Finale des Actifs')
            ax6.set_xlabel('Actif')
            ax6.set_ylabel('Poids')
            ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("STATISTIQUES D'ENTRAÎNEMENT")
    print("="*60)
    print(f"Nombre d'épisodes: {len(df)}")
    if 'portfolio_value' in df.columns:
        print(f"Valeur finale du portefeuille: {df['portfolio_value'].iloc[-1]:.2f}€")
        print(f"Performance: {(df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0] - 1) * 100:.2f}%")
    if 'total_reward' in df.columns:
        print(f"Récompense moyenne: {df['total_reward'].mean():.4f}")
        print(f"Meilleure récompense: {df['total_reward'].max():.4f}")
    print("="*60)

def analyze_action_patterns(log_file: str = "training_log.json"):
    """Analyse les patterns dans les actions prises par l'agent."""
    if not Path(log_file).exists():
        print(f"Fichier de log non trouvé: {log_file}")
        return
    
    with open(log_file, 'r') as f:
        training_data = [json.loads(line) for line in f]
    
    df = pd.DataFrame(training_data)
    
    if 'final_weights' not in df.columns:
        print("Données des poids non trouvées dans le log")
        return
    
    # Extract weight evolution
    weight_evolution = []
    for _, row in df.iterrows():
        weights = row['final_weights']
        if isinstance(weights, (list, np.ndarray)) and len(weights) > 0:
            weight_evolution.append(weights[:-1])  # Exclude cash
    
    if not weight_evolution:
        print("Aucune donnée de poids valide trouvée")
        return
    
    weight_matrix = np.array(weight_evolution)
    
    # Plot weight evolution
    plt.figure(figsize=(12, 8))
    
    for i in range(weight_matrix.shape[1]):
        plt.plot(weight_matrix[:, i], label=f'Actif {i+1}', alpha=0.7)
    
    plt.title('Évolution des Poids des Actifs')
    plt.xlabel('Épisode')
    plt.ylabel('Poids')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculate turnover
    if len(weight_matrix) > 1:
        turnovers = []
        for i in range(1, len(weight_matrix)):
            turnover = np.sum(np.abs(weight_matrix[i] - weight_matrix[i-1])) / 2
            turnovers.append(turnover)
        
        plt.figure(figsize=(10, 6))
        plt.plot(turnovers, linewidth=2, color='red')
        plt.title('Turnover par Épisode')
        plt.xlabel('Épisode')
        plt.ylabel('Turnover')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f"Turnover moyen: {np.mean(turnovers):.4f}")
        print(f"Turnover médian: {np.median(turnovers):.4f}")

def compare_with_benchmarks(results_dict: Dict):
    """Compare les résultats avec des benchmarks simples."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = ['annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown']
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        values = []
        labels = []
        for name, results in results_dict.items():
            if metric in results:
                values.append(results[metric])
                labels.append(name)
        
        bars = ax.bar(labels, values, color=colors[i], alpha=0.7)
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylabel('Valeur')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

def monitor_training_progress(log_file: str = "training_log.json", refresh_interval: int = 10):
    """Monitore l'entraînement en temps réel (à appeler périodiquement)."""
    try:
        plot_training_curves(log_file)
    except Exception as e:
        print(f"Erreur lors de la mise à jour des graphiques: {e}")

if __name__ == "__main__":
    # Exemples d'utilisation
    print("Visualisation des courbes d'entraînement...")
    plot_training_curves()
    
    print("\nAnalyse des patterns d'action...")
    analyze_action_patterns()
