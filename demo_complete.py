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
import logging
import os
import torch

from config import Config
from train import PortfolioTrainer
from evaluate_v2 import evaluate_model

# Désactiver les warnings inutiles
warnings.filterwarnings("ignore")

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def run_complete_demo():
    """Exécute une démonstration complète du système."""
    
    # Initialiser le device depuis la config
    device = Config.init_device()
    print(f"🔧 Device utilisé: {device}")
    
    print("🚀" + "="*60 + "🚀")
    print("     SAC PORTFOLIO OPTIMIZER - DÉMONSTRATION COMPLÈTE")
    print("🚀" + "="*60 + "🚀")
    print()
    print(f"⚙️  Device utilisé : {device}")
    print()
    
    # ========================================
    # PHASE 1: ENTRAÎNEMENT RAPIDE
    # ========================================
    print("📈 PHASE 1: ENTRAÎNEMENT DE L'AGENT SAC")
    print("-" * 50)
    
    config_overrides = {
        "DEVICE": device,              # <-- correction pour forcer le bon device
        "EVAL_FREQUENCY": 5,
        "SAVE_FREQUENCY": 20,
        "BATCH_SIZE": 128,
        "MAX_STEPS_PER_EPISODE": 100,  # Réduit pour test rapide
    }
    
    model_path = "models/sac_portfolio_agent.pth"
    
    try:
        trainer = PortfolioTrainer(config_overrides)
        print("🧠 Démarrage de l'entraînement (5 épisodes)...")
        
        metrics = trainer.train(num_episodes=5)
        
        print("✅ Entraînement terminé avec succès!")
        if metrics and "total_return" in metrics and metrics["total_return"]:
            final_return = metrics["total_return"][-1]
            print(f"📊 Retour final d'entraînement: {final_return:.2%}")
        
        print(f"💾 Modèle sauvegardé dans: {model_path}")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement: {e}")
        print("⚠️  Continuons avec l'évaluation d'un modèle non-entraîné...")
    
    print()
    
    # ========================================
    # PHASE 2: ÉVALUATION COMPLÈTE
    # ========================================
    print("📊 PHASE 2: ÉVALUATION DES PERFORMANCES")
    print("-" * 50)
    
    try:
        print("🔍 Évaluation sur les périodes validation et test...")
        
        # Vérifier que le modèle existe
        eval_model_path = "models/final_model.pth"
        
        results = evaluate_model(model_path=eval_model_path)
        
        if results:
            print("✅ Évaluation terminée avec succès!")
            print()
            print("📈 RÉSULTATS OBTENUS:")
            print("=" * 30)
            
            for period, metrics in results.items():
                if "Agent" in period:
                    period_name = period.replace("Agent_", "")
                    print(f"\n🎯 Période {period_name}:")
                    print(f"   • Rendement total: {metrics.get('total_return', 0):.2%}")
                    print(f"   • Rendement annualisé: {metrics.get('annualized_return', 0):.2%}")
                    print(f"   • Ratio de Sharpe: {metrics.get('sharpe_ratio', 0):.3f}")
                    print(f"   • Volatilité: {metrics.get('volatility', 0):.2%}")
                    print(f"   • Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            
            print(f"\n📊 Graphiques sauvegardés dans: results/")
            print(f"📋 Métriques détaillées dans: results/metrics_summary.csv")
            
        else:
            print("⚠️  Évaluation n'a pas retourné de résultats")
            
    except Exception as e:
        print(f"❌ Erreur lors de l'évaluation: {e}")
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
    print(f"   • {model_path} - Modèle entraîné")
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
