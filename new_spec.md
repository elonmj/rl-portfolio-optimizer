Note de Cadrage du Projet – Gestion Dynamique de Portefeuille avec SAC
1. Contexte et Objectifs

Le projet vise à concevoir un système de gestion dynamique de portefeuille d’actions pour le marché de la BRVM, capable de prendre des décisions d’investissement autonomes et adaptatives. L’objectif principal est de maximiser les rendements ajustés du risque tout en minimisant l’exposition aux pertes extrêmes et en respectant des contraintes réalistes de portefeuille.

Objectifs spécifiques :

Développer un agent d’apprentissage par renforcement utilisant l’algorithme Soft Actor-Critic (SAC).

Intégrer des indicateurs financiers avancés et des mesures de risque (CVaR, copules).

Fournir un outil capable de s’adapter aux conditions de marché changeantes et à la dynamique des meilleurs actifs sélectionnés (Top-K).

Évaluer les performances via des KPIs financiers, d’apprentissage et opérationnels.

2. Périmètre du Projet

Couverture fonctionnelle :

Sélection dynamique des K meilleurs actifs chaque période.

Calcul et intégration d’indicateurs techniques pour chaque actif (MACD, RSI, CCI, ADX, Stochastique, Bollinger, EMA, ATR, SAR, SMA, OBV, MFI…).

Construction d’un vecteur d’état complet pour l’agent :

Caractéristiques des K actifs sélectionnés.

Informations sur le portefeuille (poids actuels, liquidités, valeur totale).

Mesures de risque (CVaR, dépendance via copules).

Allocation d’actifs via l’agent SAC : poids pour chaque actif + liquidités.

Fonction de récompense prenant en compte : rendement, frais de transaction, pénalités CVaR, drawdown, asymétrie (skew), stabilité des rendements, entropie des poids.

Simulation hebdomadaire sur données historiques et apprentissage par essais et erreurs.

Exclusions :

L’agent ne sélectionne pas les actifs du marché ; la sélection Top-K est faite en amont.

Pas de prévisions de prix explicites : l’agent travaille sur les caractéristiques des actifs.

3. Description de l’Agent SAC
3.1 Observation (État)

L’agent reçoit un vecteur d’état comprenant :

Caractéristiques des K meilleurs actifs : indicateurs techniques, prix, volumes, retards (lags).

Composition du portefeuille : poids des actions, liquidités.

Risque : CVaR du portefeuille, dépendances via copules R-Vines.

3.2 Action

Allocation continue des K actifs + cash (somme des poids = 1).

Conversion des poids en actions entières, générant éventuellement des liquidités non investies.

3.3 Récompense

La fonction de récompense est composite :

Rendement du portefeuille.

Frais de transaction et coût d’opportunité du non-investissement.

Pénalités pour le CVaR, drawdown, et skew.

Bonus pour stabilité des rendements et diversification (entropie des poids).

Normalisation pour stabiliser l’apprentissage.

3.4 Apprentissage

Actor-Critic : l’acteur génère la politique, les critiques évaluent la qualité des actions.

Mise à jour progressive des réseaux critiques (soft updates).

Régularisation : dropout, layer normalization, clipping des gradients.

Learning rate avec cosine annealing et redémarrages pour exploration.

4. Processus de Gestion Dynamique

Observation et Préparation

Sélection du Top-K actif.

Calcul des indicateurs techniques et construction du vecteur d’état.

Décision

L’agent SAC propose une allocation (poids).

Action stochastique : softmax avec température décroissante pour équilibre exploration/exploitation.

Apprentissage

Récompense calculée et normalisée.

Mise à jour de la politique de l’agent.

Boucle continue chaque semaine avec adaptation aux nouveaux K actifs.

5. Indicateurs Clés de Performance (KPIs)

Financiers :

Rendement annualisé, volatilité annualisée, ratio de Sharpe, ratio de Sortino, Maximum Drawdown, rendement cumulé.

Apprentissage :

Récompense cumulée par épisode.

Opérationnels :

Taux de rotation du portefeuille (turnover), coûts de transaction.

6. Gestion du Risque

CVaR pour mesurer la perte moyenne dans les pires scénarios.

Copules R-Vines pour capturer la dépendance extrême entre actifs.

Intégration des mesures dans le vecteur d’état pour que l’agent minimise le risque extrême.

7. Contraintes et Hypothèses

Top-K actif déterminé par un processus externe.

Données historiques disponibles et nettoyées.

L’agent travaille sur un vecteur d’état de dimension fixe même si les K actifs changent.

Simulation hebdomadaire pour apprentissage.

8. Livrables

Module Python pour :

Sélection dynamique Top-K.

Calcul des indicateurs techniques.

Construction du vecteur d’état pour SAC.

Entraînement de l’agent SAC et simulation de portefeuille.

Tableau de bord des KPIs financiers et d’apprentissage.

Documentation technique et note de cadrage.

Un environnement adapté à ton projet doit inclure :

1. Les Entrées (state)

Ton vecteur d’état doit contenir tout ce que l’agent a besoin de savoir pour décider :

Indicateurs financiers par actif (rendement, volatilité, momentum, ratios techniques, etc.).

Relations de dépendance (copules R-vines) → matrice de corrélations / dépendances entre actifs.

Volatilité catégorielle (Low, Medium, High).

Portefeuille courant (poids actuels des actifs).

Valeur actuelle du portefeuille (NAV).

Mesures de risque (CVaR, drawdown, etc.).

2. Les Actions

Un vecteur d’allocation (weights) : proportion du capital sur chaque actif.

Contraintes : poids ≥ 0, somme = 1.

3. Le Reward

Tu veux un reward basé sur plusieurs dimensions :

Rendement ajusté (Sharpe-like).

Pénalité volatilité.

Pénalité drawdown.

Pénalité CVaR (risque de queue).

Bonus entropie/diversification.

Pénalité skewness (asymétrie négative).

Coûts de transaction.

4. Le Suivi (info)

Chaque semaine tu veux pouvoir suivre :

La valeur du portefeuille.

La composition du portefeuille (poids des actifs).

Les rendements semaine après semaine.

Les métriques de risque (CVaR, drawdown, volatilité).

5. Le Cycle de l’Environnement

reset() : initialise portefeuille = 1.0, allocations égales.

step(action) : applique nouvelle allocation → calcule rendement, met à jour valeur du portefeuille → calcule reward → retourne nouvel état + infos.

render() : affiche graphiques / logs de l’évolution.