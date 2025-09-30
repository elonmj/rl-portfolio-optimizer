# 📈 SAC Portfolio Optimizer - Documentation Système

##   Vue d'ensemble

Ce système implémente un optimiseur de portefeuille basé sur l'algorithme **Soft Actor-Critic (SAC)** avec mécanisme d'attention pour la gestion quantitative de portefeuilles financiers.

### 🌟 Caractéristiques principales

- **Algorithme**: Soft Actor-Critic avec attention et CVaR pour la gestion des risques
- **Assets**: Support jusqu'à 45 tickers avec allocation dynamique
- **Contraintes**: Respect des périodes de détention minimales et stratégies de buffer
- **Évaluation**: Backtesting complet avec métriques financières standardisées
- **GPU Support**: Détection automatique GPU avec fallback CPU

## 🚀 Quick Start - Test Rapide

### 1. Installation

```bash
# Cloner le repository (si applicable)
# cd OptimPortefeuille

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Données requises

Assurez-vous que le dossier `datas/` contient :
- `actions_secteurs_pays.xlsx`
- `all_datas.xlsx`  
- `dividendes.xlsx`
- `nb_actions.xlsx`

### 3. Test complet (recommandé)

```bash
# Test rapide : Entraînement (10 épisodes) + Évaluation complète
python demo_complete.py
```

### 4. Tests individuels

```bash
# Test d'entraînement uniquement (5 épisodes)
python test_training.py

# Évaluation d'un modèle existant
python evaluate_v2.py

# Entraînement complet (1000 épisodes)
python train.py
```

## 📊 Résultats attendus

Après `python demo_complete.py`, vous obtiendrez :

- **📈 Graphiques** : `results/performance_analysis.png`
- **📋 Métriques** : `results/metrics_summary.csv`
- **🤖 Modèle** : `models/sac_portfolio_agent.pth`
- **📝 Logs** : `logs/training_YYYYMMDD_HHMMSS.log`

### Métriques typiques

| Période | Rendement Annuel | Ratio Sharpe | Max Drawdown |
|---------|------------------|--------------|--------------|
| Validation (2017-2021) | +3-4% | 0.5-0.6 | -20% |
| Test (2022-2024) | Variable | Variable | -15-20% |

## 🏗️ Architecture technique

### Composants principaux

```
📦 SAC Portfolio Optimizer
├── 📊 data_processing.py - Chargement et traitement des données
├── 🌍 environment.py - Environnement de trading simulé  
├── 🧠 models.py - Réseaux de neurones avec attention
├── 🤖 agent.py - Agent SAC principal
├──   train.py - Script d'entraînement
├── 📈 evaluate_v2.py - Évaluation et backtesting
├── ⚙️ config.py - Configuration centralisée
└── 🚀 demo_complete.py - Test complet
```

### GPU vs CPU

- **GPU détecté** : Utilise automatiquement CUDA pour accélération
- **CPU seulement** : Mode fallback avec optimisations mémoire  
- **Détection automatique** : Aucune configuration manuelle requise

## ⚙️ Configuration (optionnelle)

Le système fonctionne avec les paramètres par défaut. Pour personnaliser, modifier `config.py` :

```python
class Config:
    # Périodes de données
    TRAIN_START = "1998-01-01"    # Début entraînement
    TRAIN_END = "2016-12-31"      # Fin entraînement
    TEST_START = "2022-01-01"     # Début test
    
    # Portefeuille
    INITIAL_CASH = 1_000_000      # Capital initial
    MAX_ASSETS = 12               # Nombre max d'assets
    
    # Entraînement
    MAX_EPISODES = 1000           # Episodes d'entraînement
    BATCH_SIZE = 256              # Taille de batch
    
    # Performance - Auto-détection GPU/CPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

##   Détails techniques

### Contraintes de trading (selon spec.md)

- **Période de détention minimale** : 4 semaines
- **Stratégie de buffer** : Maintenir top 11-12, remplacer si rang > 15
- **Gestion des risques** : CVaR 5% avec pondération 50%
- **Indicateurs** : 21 indicateurs techniques automatiques

### Architecture de l'agent

```python
# Dimensions pour 10 assets:
state_dim = 10*21 + 10 + 1 + 10 = 231  # Features + weights + cash + holdings
action_dim = 10                          # Nouveau portefeuille

# Mécanisme d'attention pour pondérer l'importance des assets
```

## 📊 Métriques calculées

- **Rendement total et annualisé**
- **Ratio de Sharpe / Sortino** 
- **Maximum Drawdown**
- **CVaR 5%** (Conditional Value at Risk)
- **Comparaison vs Buy & Hold**

## 🔧 Troubleshooting

### Problèmes courants

**Erreur GPU** : Le système bascule automatiquement en mode CPU
**Données manquantes** : Vérifier la structure du dossier `datas/`
**Mémoire insuffisante** : Le système s'adapte automatiquement

### Support des environnements

| Environnement | GPU | CPU | Replay Buffer | Performance |
|---------------|-----|-----|---------------|-------------|
| Local GPU     |     |     |               | Optimal     |
| Local CPU     | ❌   |     | ❌             | Dégradé     |
| Kaggle        |     |     |               | Optimal     |
| Colab         |     |     |               | Optimal     |

---

*📧 Système prêt à l'emploi - Exécuter `python demo_complete.py` pour démarrer*
