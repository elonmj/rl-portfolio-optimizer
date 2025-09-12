
#### **Document 1 : Note de Cadrage du Projet**

1.  **Titre du Projet :** Gestion Dynamique de Portefeuille basée sur le Reinforcement Learning avec Minimisation du Risque Extrême (CVaR).

2.  **Objectif Métier :** Construire un agent IA autonome qui gère un portefeuille d'actions en maximisant le rendement ajusté au risque (Ratio de Sharpe/Sortino) tout en respectant des contraintes de gestion réalistes pour minimiser les coûts de transaction et le turnover, et en contrôlant spécifiquement le risque de pertes extrêmes (CVaR).

3.  **Périmètre du Projet :**
    *   **Univers d'Actifs :** Actions et dividendes listés dans `all_datas.xlsx` et `dividendes.xlsx`.
    *   **Périodes Temporelles :**
        *   **Entraînement :** De la date minimale disponible (1998) à fin 2016.
        *   **Validation :** 2017 à 2021.
        *   **Test :** 2022 à 2024.
    *   **Contraintes de Gestion :**
        *   Pas de vente à découvert.
        *   Pas d'effet de levier (somme des poids ≤ 1).
        *   `min_holding` : 4 semaines.
        *   `buffer_keep` : Rangs 11 et 12.
        *   `replace_if` : Rang 15 ou plus.

4.  **Indicateurs Clés de Performance (KPIs) :**
    *   **Financiers :** Rendement Annualisé, Volatilité Annualisée, Ratio de Sharpe, Ratio de Sortino, **Maximum Drawdown**, Rendement Cumulé.
    *   **Apprentissage :** Récompense cumulée par épisode.
    *   **Opérationnels :** Taux de rotation du portefeuille (turnover).

---

#### **Document 2 : Spécifications Techniques et Architecturales**

#### **1. Vue d'Ensemble de l'Architecture Modulaire**

L'architecture du projet est conçue pour la clarté, la maintenabilité et la séparation des responsabilités. Chaque fichier a un rôle unique :

```
/portfolio_rl/
├── config.py                 # Le "tableau de bord" : tous les paramètres.
├── data_processing.py        # La "cuisine" des données : chargement et préparation.
├── environment.py            # Le "simulateur" : l'environnement de marché avec ses règles.
├── models.py                 # Les "cerveaux" : les architectures des réseaux de neurones.
├── agent.py                  # Le "décideur" : la logique de l'agent SAC qui utilise les cerveaux.
├── train.py                  # Le "chef d'orchestre" : le script qui lance et gère l'entraînement.
└── evaluate.py               # L'"analyste" : le script pour le backtesting et la visualisation.
```

---

#### **1-2. Flux de Données et de Logique**

1.  **Initialisation (`train.py`)**: Lit `config.py`, instancie les modules de données, l'environnement `PortfolioEnv` et l'agent `SACAgent`.
2.  **Boucle d'Entraînement (`train.py`)**:
    a. L'agent observe l'**état** (`state`) fourni par l'environnement.
    b. L'agent produit une **action** (vecteur de poids cibles) via `agent.select_action(state)`.
    c. L'environnement exécute cette action via `env.step(action)`, appliquant les règles métier.
    d. La transition `(état, action, récompense, nouvel_état, terminé)` est stockée.
    e. L'agent met à jour ses réseaux via `agent.update()`.

---



2.  **Algorithme Fondamental :** Soft Actor-Critic (SAC) basé sur le framework de l'**entropie maximale**.

3.  **Solution Architecturale :**
    *   **Mécanisme d'Attention :** Une couche d'**auto-attention (self-attention)** est utilisée pour l'Acteur et les Critiques. Pour le tenseur d'indicateurs des actifs `(K, 21)`, les matrices de **Requête (Query)**, **Clé (Key)** et **Valeur (Value)** sont toutes dérivées de ce même tenseur d'entrée.
    *   **Implémentation des Contraintes :** Les règles métier (`min_holding`, `buffer_keep`, `replace_if`) sont implémentées au sein de la fonction `step` de l'environnement `PortfolioEnv`.

---

#### **Document 3 : Plan de Gestion des Données**

*Implémentation dans `data_processing.py`*

1.  **Classe `DataHandler` & Structure des Fichiers :**
    *   `all_datas.xlsx` : Fichier Excel multi-feuilles (nom de feuille = Ticker). Colonnes : `Date`, `Close`, `High`, `Low`, `Volume`.
    *   `dividendes.xlsx` : Feuille unique. Colonnes : `Date` (année), Tickers...

2.  **Classe `FeatureProcessor` :**
    *   **Logique de Sélection Top-K :**
        *   Formule de Score : `Score = 0.3*Rank(Momentum) - 0.3*Rank(Volatilité) + 0.3*Rank(Liquidité) + 0.1*Rank(Dividende)` sur 12 semaines.
        *   `Momentum`: `(dernier_prix / premier_prix) - 1`.
        *   `Volatilité`: `df['Close'].pct_change().std()`.
        *   `Liquidité`: `df['Volume'].mean()`.
        *   `Dividende`: `dividende_N-1 / dernier_prix`.
    *   **Liste des 21 Indicateurs Techniques :** MACD, RSI, CCI, Stochastic Oscillator, Williams %R, Bandes de Bollinger (Sup, Milieu, Inf), EMA(10), ATR(14), Parabolic SAR, Lags de Clôture (1, 3, 5), SMA (50, 20), OBV, MFI(14), Volatility Index(20), Volume, et Prix de Clôture.
    *   **Normalisation :** Tous les indicateurs (sauf le prix de clôture) sont normalisés sur la période d'entraînement.


*   **Rôle de l'Agent vs. Rôle de l'Environnement :**
    *   **L'Agent (Le Stratège) :** Observe l'état du marché et de son portefeuille, puis propose une **allocation de capital CIBLE**.
    *   **L'Environnement (L'Exécutant) :** Reçoit cette allocation cible, applique les règles de gestion, puis exécute les transactions possibles.

#### **Document 4 : Conception du Modèle RL (MDP)**

1.  **Espace d'États :** Un tuple contenant (1) Tenseur d'indicateurs `(K, 21)`, (2) Vecteur de poids actuels `(K)`, et (3) Vecteur global `(2)` (cash, capital).

2.  **Espace d'Actions :** Un vecteur continu de taille `k+1` représentant les poids des `k` actifs plus la liquidité.

3.  **Fonction de Récompense :** `reward_t = R_t - λ * CVaR_α - transaction_cost`
    *   `R_t` : Rendement net pondéré du portefeuille : `Σ (wi * ri,t)`.
    *   `CVaR_α` : **Conditional Value at Risk**, calculé sur une fenêtre glissante des N dernières semaines (défini dans `config.py`) des rendements hebdomadaires du portefeuille.
    *   `transaction_cost` : Coût des transactions : `COÛT_PCT * valeur_portefeuille * Σ |w_i_nouveau - w_i_actuel|`.

4.  **Logique de la Fonction `step(action)` :**
    *   **Étape 1 : Avancer le Temps & Mettre à jour les Compteurs.** Passer à la semaine `t+1`. Incrémenter `holding_counts` de 1 pour tous les actifs détenus.
    *   **Étape 2 : Décisions de Vente.** Identifier la nouvelle liste Top-K. Pour chaque actif en portefeuille, vendre si les règles (`replace_if`, `buffer_keep`, `min_holding`) le forcent. Mettre à jour `self.cash`.
    *   **Étape 3 : Décisions d'Achat.** Utiliser les poids cibles (`action`) pour allouer le **cash disponible** aux nouveaux actifs du Top-K.
    *   **Étape 4 : Exécution.** Mettre à jour `self.portfolio`, `self.cash`, et initialiser `holding_counts` à 1 pour les nouveaux actifs.
    *   **Étape 5 : Retour.** Calculer la récompense et le nouvel état pour la période `t+1`.

---

#### **Document 5 : Architecture et Implémentation du Modèle**

##### **5.1. Fichier de Configuration (`config.py`)**
```python
# --- Paramètres de l'environnement et des données ---
INITIAL_CAPITAL = 10000.0; K_ASSETS = 10; MIN_HOLDING_WEEKS = 4
BUFFER_KEEP_RANKS = [11, 12]; REPLACE_IF_RANK = 15
TRANSACTION_COST_PCT = 0.001; RISK_AVERSION_LAMBDA = 0.5; CVAR_ALPHA = 0.05
CVAR_WINDOW_WEEKS = 52 # Fenêtre pour le calcul du CVaR historique
TRAIN_START_DATE = "1998-01-01"; TRAIN_END_DATE = "2016-12-31"
VALIDATION_START_DATE = "2017-01-01"; VALIDATION_END_DATE = "2021-12-31"
TEST_START_DATE = "2022-01-01"; TEST_END_DATE = "2024-12-31"

# --- Hyperparamètres du Modèle et de l'Agent SAC ---
STATE_DIM_INDICATORS = 21; HIDDEN_DIM = 256; LEARNING_RATE = 3e-4
GAMMA = 0.99; TAU = 0.005; ALPHA = 0.2; BUFFER_SIZE = 100000; BATCH_SIZE = 64
```

##### **5.2. Architecture des Réseaux (`models.py`)**
*   Implémenter `ActorWithAttention` et `CriticWithAttention`. Il n'y aura pas de réseau de Valeur (`ValueNetwork`) séparé.
*   **Mathématique (Acteur) :** L'Acteur utilise `tanh` et corrige la log-probabilité : `log πφ(at|st) = log μ(ut|st) - Σ log(1 - tanh²(ui))`.

##### **5.3. Fonctions de Perte de SAC (`agent.py`)**
La méthode `update()` implémente les fonctions de perte de la version de SAC sans réseau de valeur explicite :
1.  **Calcul de la Q-Cible :** `y(r, st+1) = r + γ * (min(Q'_θ1, Q'_θ2) - α log πφ(at+1|st+1))`
2.  **Loss des Critiques (Qθ) :** `J_Q(θ) = E[ (Qθ(st, at) - y)^2 ]`
3.  **Loss de la Politique (Acteur πφ) :** `J_π(φ) = E[ α log πφ(at|st) - min(Qθ1, Qθ2) ]`

---

#### **Document 6 : Plan de Déploiement et de Maintenance**

1.  **Versionnement :** Utiliser **Git**.
2.  **Dépendances :** Maintenir un fichier `requirements.txt`.
3.  **Tests :** Implémenter des tests unitaires avec `pytest`.
4.  **Logging :** Utiliser le module `logging`.
5.  **Ré-entraînement :** Prévoir une stratégie pour le fine-tuning périodique.

---

#### **Document 7 : Plan d'Exécution et de Décomposition des Tâches**

1.  **Phase 1 :** Mise en place de l'architecture de base (fichiers, `requirements.txt`, `config.py`).
2.  **Phase 2 : Implémentation du Cœur**
    *   **Tâche 2.1 :** Coder `DataHandler` et `FeatureProcessor`.
    *   **Tâche 2.2 :** Coder `PortfolioEnv`, en implémentant la logique de `step`.
3.  **Phase 3 : Implémentation de l'Agent**
    *   **Tâche 3.1 :** Coder `ActorWithAttention` et `CriticWithAttention`.
    *   **Tâche 3.2 :** Coder `ReplayBuffer` et `SACAgent`, en implémentant les fonctions de perte.
4.  **Phase 4 & 5 :** Orchestration et Évaluation
    *   **Tâche 4.1 :** Écrire le script `train.py`.
    *   **Tâche 5.1 :** Écrire le script `evaluate.py`, en y incluant le calcul du Maximum Drawdown.