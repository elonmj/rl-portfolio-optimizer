"""
🚀 DEMO COMPLET - SAC Portfolio Optimizer
=======================================

Script de démonstration complète qui :
1. Entraîne l'agent SAC sur 10 épisodes (test rapide)
2. Évalue les performances sur les périodes de validation et test
3. Affiche les résultats et génère les graphiques

Utilisation: python demo_complete.py
"""

import warnings
warnings.filterwarnings('ignore')

from train import PortfolioTrainer
from evaluate_v2 import evaluate_model
import logging

# Configuration du logging pour un affichage propre
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_complete_demo():
    """Exécute une démonstration complète du système."""
    
    print("🚀" + "="*60 + "🚀")
    print("     SAC PORTFOLIO OPTIMIZER - DÉMONSTRATION COMPLÈTE")
    print("🚀" + "="*60 + "🚀")
    print()
    
    # ========================================
    # PHASE 1: ENTRAÎNEMENT RAPIDE
    # ========================================
    print("📈 PHASE 1: ENTRAÎNEMENT DE L'AGENT SAC")
    print("-" * 50)
    
    # Configuration pour test rapide (10 épisodes)
    config_overrides = {
        'MAX_EPISODES': 10,
        'EVAL_FREQUENCY': 5,
        'SAVE_FREQUENCY': 20,
        'BATCH_SIZE': 128,
        'MAX_STEPS_PER_EPISODE': 100,  # Réduire pour test rapide
    }
    
    try:
        trainer = PortfolioTrainer(config_overrides)
        print('🧠 Démarrage de l\'entraînement (10 épisodes)...')
        
        metrics = trainer.train(num_episodes=10)
        
        print('✅ Entraînement terminé avec succès!')
        if metrics and 'total_return' in metrics and metrics['total_return']:
            final_return = metrics['total_return'][-1]
            print(f'📊 Retour final d\'entraînement: {final_return:.2%}')
        
        print('💾 Modèle sauvegardé dans: models/sac_portfolio_agent.pth')
        
    except Exception as e:
        print(f'❌ Erreur lors de l\'entraînement: {e}')
        print('Continuons avec l\'évaluation d\'un modèle non-entraîné...')
    
    print()
    
    # ========================================
    # PHASE 2: ÉVALUATION COMPLÈTE
    # ========================================
    print("📊 PHASE 2: ÉVALUATION DES PERFORMANCES")
    print("-" * 50)
    
    try:
        print('🔍 Évaluation sur les périodes validation et test...')
        
        # Évaluation avec replay buffer si GPU disponible
        results = evaluate_model(use_replay_buffer=True)
        
        if results:
            print('✅ Évaluation terminée avec succès!')
            print()
            print("📈 RÉSULTATS OBTENUS:")
            print("=" * 30)
            
            for period, metrics in results.items():
                if 'Agent' in period:
                    period_name = period.replace('Agent_', '')
                    print(f"\n🎯 Période {period_name}:")
                    print(f"   • Rendement total: {metrics.get('total_return', 0):.2%}")
                    print(f"   • Rendement annualisé: {metrics.get('annualized_return', 0):.2%}")
                    print(f"   • Ratio de Sharpe: {metrics.get('sharpe_ratio', 0):.3f}")
                    print(f"   • Volatilité: {metrics.get('volatility', 0):.2%}")
                    print(f"   • Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            
            print(f"\n📊 Graphiques sauvegardés dans: results/")
            print(f"📋 Métriques détaillées dans: results/metrics_summary.csv")
            
        else:
            print('⚠️  Évaluation n\'a pas retourné de résultats')
            
    except Exception as e:
        print(f'❌ Erreur lors de l\'évaluation: {e}')
        import traceback
        traceback.print_exc()
    
    print()
    
    # ========================================
    # CONCLUSION
    # ========================================
    print("🎉" + "="*60 + "🎉")
    print("               DÉMONSTRATION TERMINÉE")
    print("🎉" + "="*60 + "🎉")
    print()
    print("📁 Fichiers générés:")
    print("   • models/sac_portfolio_agent.pth - Modèle entraîné")
    print("   • results/performance_analysis.png - Graphiques de performance")
    print("   • results/metrics_summary.csv - Métriques détaillées")
    print("   • logs/ - Logs d'entraînement")
    print()
    print("🔄 Pour un entraînement complet (1000 épisodes):")
    print("   python train.py")
    print()
    print("📊 Pour évaluation uniquement:")
    print("   python evaluate_v2.py")
    print()

if __name__ == "__main__":
    run_complete_demo()
