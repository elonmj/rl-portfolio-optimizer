#!/usr/bin/env python3
"""
Script de lancement d'évaluation sur Kaggle
Utilise kaggle_manager pour exécuter l'évaluation complète
"""

import os
import sys
import json
from datetime import datetime

# Import kaggle_manager
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'kaggle'))
    from kaggle_manager_github import KaggleManagerGitHub as KaggleManager
    print("✅ KaggleManagerGitHub importé avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import KaggleManagerGitHub: {e}")
    sys.exit(1)


def main():
    """
    Lance l'évaluation complète sur Kaggle
    """
    print("🎯 LANCEMENT DE L'ÉVALUATION SUR KAGGLE")
    print("=" * 50)
    
    try:
        # Import de la fonction directe
        from kaggle_manager_github import run_kaggle_evaluation
        
        print(f"� Lancement de l'évaluation avec run_kaggle_evaluation...")
        
        # Lancer l'évaluation
        success = run_kaggle_evaluation(
            model_path="models/sac_portfolio_agent_kaggle.pth",
            repo_url="https://github.com/elonmj/rl-portfolio-optimizer.git", 
            branch="feature/training-config-updates",
            timeout=3600
        )
        
        if success:
            print(f"✅ Évaluation Kaggle terminée avec succès!")
            
            # Vérifier s'il y a des fichiers téléchargés
            downloaded_files = []
            for pattern in ['*.png', '*.json', '*.md', '*.csv']:
                import glob
                files = glob.glob(pattern)
                downloaded_files.extend(files)
            
            if downloaded_files:
                print(f"📁 Fichiers résultats disponibles:")
                for f in downloaded_files:
                    print(f"   - {f}")
                
                create_evaluation_summary(downloaded_files)
            else:
                print(f"⚠️ Aucun fichier de résultat trouvé localement")
                print(f"💡 Vérifiez manuellement les outputs sur Kaggle")
        
        else:
            print(f"❌ Échec de l'évaluation Kaggle")
            return 1
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def create_evaluation_summary(result_files):
    """
    Crée un résumé des résultats d'évaluation
    """
    print(f"\n📋 Création du résumé d'évaluation...")
    
    summary = []
    summary.append("# 🎯 RÉSUMÉ DE L'ÉVALUATION KAGGLE")
    summary.append("=" * 50)
    summary.append(f"📅 Évaluation terminée le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("")
    
    summary.append("## 📁 Fichiers générés")
    for file_path in result_files:
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        summary.append(f"- `{file_name}`: {file_size:,} bytes")
    summary.append("")
    
    # Analyser les données JSON si disponibles
    json_files = [f for f in result_files if f.endswith('.json')]
    if json_files:
        try:
            with open(json_files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'evaluation_metadata' in data:
                meta = data['evaluation_metadata']
                summary.append("## 📊 Métadonnées d'évaluation")
                summary.append(f"- **Assets évalués**: {meta.get('n_assets', 'N/A')}")
                summary.append(f"- **Épisodes**: {meta.get('n_episodes', 'N/A')}")
                summary.append(f"- **Dimensions état**: {meta.get('observation_dim', 'N/A')}")
                summary.append("")
            
            if 'episodes' in data and data['episodes']:
                episodes = data['episodes']
                avg_return = sum(ep['metrics']['total_return'] for ep in episodes) / len(episodes)
                avg_sharpe = sum(ep['metrics']['sharpe_ratio'] for ep in episodes) / len(episodes)
                
                summary.append("## 🎯 Résultats clés")
                summary.append(f"- **Rendement moyen**: {avg_return:.2%}")
                summary.append(f"- **Sharpe moyen**: {avg_sharpe:.3f}")
                summary.append("")
        
        except Exception as e:
            summary.append(f"⚠️ Erreur d'analyse JSON: {e}")
            summary.append("")
    
    summary.append("## 🎉 Évaluation terminée avec succès!")
    summary.append("Tous les fichiers ont été téléchargés depuis Kaggle.")
    
    with open('evaluation_kaggle_summary.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
    
    print("   ✅ Résumé sauvegardé: evaluation_kaggle_summary.md")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)