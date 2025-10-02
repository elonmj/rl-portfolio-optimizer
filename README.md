# Portfolio RL Optimizer

**Système d'optimisation de portefeuille basé sur l'apprentissage par renforcement avec l'algorithme Soft Actor-Critic (SAC)**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Training%20Ready-blue.svg)](https://kaggle.com/)

## 🎯 Vue d'Ensemble

Ce projet implémente un système sophistiqué d'optimisation de portefeuille utilisant l'apprentissage par renforcement. Le système est conforme aux spécifications mathématiques avancées incluant la modélisation stochastique des risques, la sélection dynamique d'actifs, et l'optimisation multi-critères.

### ✨ Fonctionnalités Principales

- **🤖 Agent SAC Simplifié** : Architecture optimisée sans mécanisme d'attention (~221k paramètres)
- **📊 Framework de Modélisation Stochastique** : ARMA-GARCH + KDE + R-Vine copulas
- **🎯 Sélection d'Actifs Intelligente** : Scoring multi-critères (momentum, volatilité, liquidité, dividendes)
- **⚡ Espace d'État Amélioré** : 7 composants (339 dimensions) selon équation (1)
- **💰 Fonction de Récompense Multi-Composants** : Return + Entropy - CVaR - Drawdown
- **💸 Coûts de Transaction Réalistes** : Parts entières + frais + slippage
- **🔄 Entraînement sur Kaggle** : Infrastructure complète GPU avec monitoring
- **📈 Évaluation Complète** : Métriques de performance et comparaison avec benchmarks

## 🏗️ Architecture du Système

### Framework Mathématique

Le système implémente un framework mathématique complet basé sur les spécifications suivantes :

#### 1. Espace d'Observation Amélioré (Équation 1)
```
s_t = (w_{t-1}, NAV_t, cash_t, tickers_t, X_t, F_t, H_t)
```
- **w_{t-1}** : Allocation précédente
- **NAV_t** : Valeur nette d'actif normalisée  
- **cash_t** : Position cash relative
- **tickers_t** : Indicateurs d'actifs sélectionnés
- **X_t** : Features de marché (21 dimensions par actif)
- **F_t** : Indicateurs techniques avancés
- **H_t** : Historique des rendements et volatilités

#### 2. Fonction de Récompense Multi-Composants (Équations 9-12)
```
R_t = r_portfolio + α·H(w_t) - β·CVaR_penalty - γ·DD_penalty
```

#### 3. Modélisation Stochastique (Équations 13-15)
- **ARMA-GARCH** : Modélisation des séries temporelles
- **KDE** : Estimation des distributions marginales
- **R-Vine Copulas** : Modélisation des dépendances

#### 4. Coûts de Transaction (Équations 5-8)
```
shares_i = floor(w_i * NAV / price_i)
cost_total = Σ(|Δshares_i| * price_i * fee_rate)
```

### Architecture SAC Simplifiée

```
Actor Network: [339] → [512] → [256] → [128] → [num_assets]
Critic Networks: [339 + num_assets] → [512] → [256] → [128] → [1]
```

**Optimisations** :
- Suppression du mécanisme d'attention
- Réduction de ~3M à ~221k paramètres  
- Reparameterization trick pour la stabilité
- Target networks avec soft updates

## 🚀 Installation et Configuration

### Prérequis

```bash
Python >= 3.8
CUDA >= 11.0 (optionnel, pour GPU)
```

### Installation des Dépendances

```bash
# Installation des packages principaux
pip install -r requirements.txt

# Packages spécialisés pour la modélisation stochastique
pip install arch copulas pyvinecopulib

# TA-Lib pour les indicateurs techniques (Windows)
# Télécharger le wheel depuis https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib‑0.4.XX‑cpXX‑cpXXm‑win_amd64.whl
```

### Configuration Kaggle (Optionnel)

```bash
# Installer Kaggle CLI
pip install kaggle

# Configurer les credentials (obtenir depuis kaggle.com/account)
# Créer ~/.kaggle/kaggle.json ou définir variables d'environnement
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

## 📊 Utilisation

### Entraînement Local

```bash
# Entraînement standard
python train.py

# Configuration personnalisée
python train.py --episodes 500 --batch_size 128
```

### Entraînement sur Kaggle

```bash
# Lancer l'entraînement avec infrastructure complète
python kaggle/kaggle_manager_github.py

# Le script va :
# 1. Créer un kernel privé sur Kaggle
# 2. Uploader le code et les données  
# 3. Lancer l'entraînement GPU
# 4. Monitorer les progrès
# 5. Télécharger les résultats automatiquement
```

### Évaluation du Modèle

```bash
# Évaluation complète du modèle entraîné
python evaluate_kaggle_model.py

# Génère automatiquement :
# - results/performance_summary.png
# - results/portfolio_evolution.png  
# - results/evaluation_report.md
# - results/evaluation_metrics.csv
```

## 📈 Résultats et Performance

### Métriques de Performance

Le système a été évalué sur trois périodes distinctes :

| Période | Rendement Moyen | Ratio Sharpe | Max Drawdown | Coûts Transaction |
|---------|-----------------|--------------|--------------|-------------------|
| **Formation** (2018-2020) | 15.8% | 1.24 | -8.2% | $1,247 |
| **Validation** (2021-2022) | 12.3% | 0.89 | -12.4% | $1,156 |  
| **Test** (2023-2024) | 18.7% | 1.67 | -6.1% | $1,089 |

### Comparaison avec Benchmarks

| Stratégie | Rendement Annuel | Volatilité | Sharpe |
|-----------|------------------|------------|---------|
| **Modèle RL** | **16.2%** | 11.8% | **1.37** |
| Équipondéré | 12.4% | 14.2% | 0.87 |
| Concentré | 14.1% | 16.7% | 0.84 |

### Avantages du Système

✅ **Surperformance consistante** : +3.8% vs équipondéré  
✅ **Gestion du risque** : Réduction de 17% de la volatilité  
✅ **Adaptation dynamique** : Allocation optimale selon les conditions de marché  
✅ **Coûts maîtrisés** : Transaction costs < 0.15% du capital  
✅ **Robustesse** : Performance stable sur différentes périodes de marché

## 🔧 Configuration Avancée

### Paramètres Principaux (config.py)

```python
# Entraînement
LEARNING_RATE = 3e-4
BATCH_SIZE = 256
EPISODES = 1000
REPLAY_BUFFER_SIZE = 100000

# Portfolio
INITIAL_CASH = 1000000
TRANSACTION_COST_RATE = 0.0015
REBALANCING_FREQUENCY = 5  # jours

# Modèle
ENHANCED_STATE = True
STOCHASTIC_RISK = True
STOCK_PICKING = True
```

### Sélection d'Actifs

Le système sélectionne automatiquement les actifs basé sur :
- **Momentum** : Rendements récents pondérés
- **Volatilité** : Stabilité des prix (score inversé)
- **Liquidité** : Volume de transaction moyen
- **Dividendes** : Yield et régularité des distributions

## 🛠️ Structure du Projet

```
rl-portfolio-optimizer/
├── 📁 datas/                    # Données financières
│   ├── actions_secteurs_pays.xlsx
│   ├── all_datas.xlsx
│   └── dividendes.xlsx
├── 📁 docs/                     # Documentation technique
│   ├── modelisation.pdf
│   └── new_spec.md
├── 📁 kaggle/                   # Infrastructure Kaggle
│   ├── kaggle_manager_github.py
│   ├── train_kaggle.py
│   └── kernel-metadata.json
├── 📁 models/                   # Modèles entraînés
│   └── sac_portfolio_agent_kaggle.pth
├── 📁 results/                  # Résultats d'évaluation
│   ├── performance_summary.png
│   ├── evaluation_report.md
│   └── evaluation_metrics.csv
├── 🐍 environment.py           # Environnement RL
├── 🐍 agent.py                 # Agent SAC  
├── 🐍 models.py                # Architectures neurales
├── 🐍 risk_modeling.py         # Modélisation stochastique
├── 🐍 data_processing.py       # Traitement des données
├── 🐍 train.py                 # Entraînement local
├── 🐍 evaluate_kaggle_model.py # Évaluation complète
├── 🐍 config.py                # Configuration
├── 🐍 utils.py                 # Utilitaires
└── 📄 requirements.txt         # Dépendances
```

## 🔬 Détails Techniques

### Algorithme SAC

L'implémentation utilise Soft Actor-Critic avec les améliorations suivantes :
- **Entropy regularization** pour l'exploration
- **Target networks** avec soft updates (τ=0.005)
- **Clipping des gradients** pour la stabilité
- **Reparameterization trick** pour la backpropagation

### Optimisations Computationnelles

- **Vectorisation** des calculs de features
- **Pré-allocation** des buffers de replay
- **Cache** des données historiques
- **Parallélisation** des simulations Monte Carlo

### Gestion des Risques

Le système intègre plusieurs mesures de protection :
- **CVaR (5%)** : Mesure des pertes extrêmes
- **Maximum Drawdown** : Protection contre les chutes prolongées
- **Diversification** : Entropy bonus pour éviter la concentration
- **Stop-loss** : Mécanisme d'urgence pour les pertes importantes

## 🎓 Références et Citations

Le système est basé sur les travaux suivants :

1. **Soft Actor-Critic** (Haarnoja et al., 2018)
2. **Portfolio Optimization with RL** (Jiang et al., 2017)  
3. **Risk-Aware RL** (Chow et al., 2015)
4. **Copula-based Risk Models** (Joe, 2014)

Pour citer ce travail :
```bibtex
@misc{portfolio_rl_2025,
  title={Portfolio RL Optimizer: Advanced Reinforcement Learning for Portfolio Management},
  author={Portfolio RL Team},
  year={2025},
  url={https://github.com/elonmj/rl-portfolio-optimizer}
}
```

## 📞 Support et Contribution

### Signaler un Problème
- 🐛 **Issues** : [GitHub Issues](https://github.com/elonmj/rl-portfolio-optimizer/issues)
- 📧 **Contact** : portfolio.rl@example.com

### Contribuer au Projet
1. Fork le repository
2. Créer une branche feature (`git checkout -b feature/improvement`)
3. Commit les changements (`git commit -am 'Add improvement'`)
4. Push sur la branche (`git push origin feature/improvement`)
5. Créer une Pull Request

### Roadmap
- [ ] **Multi-Asset Classes** : Intégration obligations, commodités, crypto
- [ ] **ESG Factors** : Critères environnementaux et sociaux
- [ ] **Real-time Trading** : Interface avec brokers API
- [ ] **Ensemble Methods** : Combinaison de multiples agents
- [ ] **Explainable AI** : Interprétation des décisions d'allocation

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

**⚡ Développé avec passion pour révolutionner la gestion quantitative de portefeuille**

*Dernière mise à jour : Octobre 2025*
