"""
Environnement RL pour la gestion de portefeuille avec CVaR et contraintes de trading.
Implémente PortfolioEnv selon spec.md.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from config import Config
from data_processing import DataHandler, FeatureProcessor

class PortfolioEnv(gym.Env):
    """
    Environnement RL pour la gestion de portefeuille avec:
    - CVaR pour la gestion du risque
    - Contraintes de trading (min_holding, buffer_keep, replace_if)
    - Pas de short selling ni de levier
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 tickers: List[str],
                 start_date: str,
                 end_date: str,
                 initial_cash: float = Config.INITIAL_CASH,
                 max_assets: int = Config.MAX_ASSETS,
                 data_handler: DataHandler = None,
                 feature_processor: FeatureProcessor = None):
        
        super().__init__()
        
        # Configuration
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.max_assets = max_assets
        
        # Contraintes de trading selon spec.md
        self.min_holding_weeks = Config.MIN_HOLDING_WEEKS
        self.buffer_keep = Config.BUFFER_KEEP
        self.replace_if = Config.REPLACE_IF
        
        # CVaR parameters
        self.cvar_alpha = Config.CVAR_ALPHA
        self.cvar_lambda = Config.CVAR_LAMBDA
        self.cvar_window = Config.CVAR_WINDOW
        
        # Data processing
        self.data_handler = data_handler or DataHandler()
        self.feature_processor = feature_processor or FeatureProcessor(self.data_handler)
        
        # Charger et préparer les données
        self._prepare_data()
        
        # États de l'environnement
        self.current_step = 0
        self.cash = initial_cash
        self.portfolio_weights = np.zeros(len(self.valid_tickers))
        self.portfolio_positions = np.zeros(len(self.valid_tickers))  # Nombre d'actions
        self.holding_periods = np.zeros(len(self.valid_tickers))  # Durée de détention en semaines
        
        # Historique pour CVaR
        self.returns_history = []
        self.portfolio_values_history = [initial_cash]
        
        # Définir les espaces d'action et d'observation
        self._define_spaces()
        
        print(f"✅ Environnement initialisé avec {len(self.valid_tickers)} assets sur {len(self.dates)} périodes")
    
    def _prepare_data(self):
        """Prépare les données pour l'environnement"""
        # Charger toutes les données si pas déjà fait
        if not hasattr(self.data_handler, 'tickers_data') or not self.data_handler.tickers_data:
            self.data_handler.load_all_data()
        
        # Préparer la matrice de features
        self.features_matrix, self.valid_tickers, self.dates = self.feature_processor.prepare_features_matrix(
            self.tickers, self.start_date, self.end_date
        )
        
        # Récupérer les prix pour le calcul des rendements
        self.prices_matrix = self._prepare_prices_matrix()
        
        # Récupérer les dividendes
        self.dividends_data = self._prepare_dividends_data()
        
    def _prepare_prices_matrix(self) -> np.ndarray:
        """Prépare la matrice des prix (Close) alignée avec les features"""
        prices_list = []
        
        for ticker in self.valid_tickers:
            ticker_data = self.data_handler.get_ticker_data(ticker, self.start_date, self.end_date)
            # Aligner avec les dates communes
            aligned_data = ticker_data[ticker_data['Date'].isin(self.dates)].sort_values('Date')
            prices_list.append(aligned_data['Close'].values)
        
        return np.column_stack(prices_list)  # Shape: (T, K)
    
    def _prepare_dividends_data(self) -> np.ndarray:
        """Prépare les données de dividendes"""
        # Pour simplifier, on assume des dividendes annuels
        # Dans une implémentation complète, il faudrait aligner les dividendes avec les dates
        dividends_matrix = np.zeros((len(self.dates), len(self.valid_tickers)))
        
        if self.data_handler.dividends_data is not None:
            for i, ticker in enumerate(self.valid_tickers):
                if ticker in self.data_handler.dividends_data.columns:
                    # Approximation: dividende moyen réparti sur l'année
                    mean_dividend = self.data_handler.dividends_data[ticker].mean()
                    if not np.isnan(mean_dividend):
                        # Répartir le dividende annuel sur 52 semaines
                        weekly_dividend = mean_dividend / 52
                        dividends_matrix[:, i] = weekly_dividend
        
        return dividends_matrix
    
    def _define_spaces(self):
        """Définit les espaces d'action et d'observation"""
        num_assets = len(self.valid_tickers)
        num_features = len(Config.TECHNICAL_INDICATORS)
        
        # Espace d'observation: features + état du portefeuille
        # Features: (K, 21), Portfolio state: (K,), Cash: (1,), Holdings: (K,)
        obs_dim = num_assets * num_features + num_assets + 1 + num_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # Espace d'action: poids du portefeuille (entre 0 et 1, somme <= 1)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(num_assets,), 
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Remet l'environnement à l'état initial"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.cash = self.initial_cash
        self.portfolio_weights = np.zeros(len(self.valid_tickers))
        self.portfolio_positions = np.zeros(len(self.valid_tickers))
        self.holding_periods = np.zeros(len(self.valid_tickers))
        
        self.returns_history = []
        self.portfolio_values_history = [self.initial_cash]
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Exécute une action dans l'environnement"""
        
        # Normaliser l'action pour que la somme soit <= 1
        action = np.clip(action, 0, 1)
        if action.sum() > 1:
            action = action / action.sum()
        
        # Appliquer les contraintes de trading
        constrained_action = self._apply_trading_constraints(action)
        
        # Calculer les trades nécessaires
        trades = self._calculate_trades(constrained_action)
        
        # Exécuter les trades
        self._execute_trades(trades)
        
        # Mettre à jour les holding periods
        self._update_holding_periods()
        
        # Avancer d'un pas temporel
        self.current_step += 1
        
        # Calculer le rendement et la valeur du portefeuille
        portfolio_return = self._calculate_portfolio_return()
        portfolio_value = self._calculate_portfolio_value()
        
        # Calculer la reward avec CVaR
        reward = self._calculate_reward(portfolio_return, portfolio_value)
        
        # Mettre à jour l'historique
        self.returns_history.append(portfolio_return)
        self.portfolio_values_history.append(portfolio_value)
        
        # Vérifier si l'épisode est terminé
        terminated = self.current_step >= len(self.dates) - 1
        truncated = False
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _apply_trading_constraints(self, action: np.ndarray) -> np.ndarray:
        """Applique les contraintes de trading selon spec.md"""
        constrained_action = action.copy()
        
        # 1. Respecter min_holding_weeks
        for i in range(len(self.valid_tickers)):
            if (self.portfolio_weights[i] > 0 and 
                self.holding_periods[i] < self.min_holding_weeks and
                constrained_action[i] < self.portfolio_weights[i]):
                # Maintenir la position si holding period < min_holding_weeks
                constrained_action[i] = self.portfolio_weights[i]
        
        # 2. Appliquer buffer_keep strategy
        current_top_positions = np.argsort(self.portfolio_weights)[::-1]
        top_11_12 = current_top_positions[:12]  # Top 12 positions actuelles
        
        # Garder les positions 11-12 (indices 10 et 11 dans le top 12)
        if len(top_11_12) >= 11:
            for idx in [10, 11]:  # Positions 11 et 12
                if idx < len(top_11_12):
                    asset_idx = top_11_12[idx]
                    if self.portfolio_weights[asset_idx] > 0:
                        # Maintenir la position dans le buffer
                        constrained_action[asset_idx] = max(
                            constrained_action[asset_idx], 
                            self.portfolio_weights[asset_idx] * 0.5  # Au moins 50% de la position
                        )
        
        # 3. Appliquer replace_if rule
        for i in range(len(self.valid_tickers)):
            if self.portfolio_weights[i] > 0:
                current_rank = np.where(current_top_positions == i)[0]
                if len(current_rank) > 0 and current_rank[0] >= self.replace_if:
                    # Réduire la position si rang > replace_if
                    constrained_action[i] = min(constrained_action[i], self.portfolio_weights[i] * 0.5)
        
        # 4. Limiter le nombre total d'assets à max_assets
        if np.sum(constrained_action > 0.01) > self.max_assets:  # Seuil minimum 1%
            # Garder seulement les top max_assets
            top_indices = np.argsort(constrained_action)[::-1][:self.max_assets]
            new_action = np.zeros_like(constrained_action)
            new_action[top_indices] = constrained_action[top_indices]
            constrained_action = new_action
        
        # Renormaliser pour que la somme soit <= 1
        if constrained_action.sum() > 1:
            constrained_action = constrained_action / constrained_action.sum()
        
        return constrained_action
    
    def _calculate_trades(self, target_weights: np.ndarray) -> np.ndarray:
        """Calcule les trades nécessaires pour atteindre les poids cibles"""
        current_prices = self.prices_matrix[self.current_step]
        
        # Calculer la valeur actuelle du portefeuille
        current_portfolio_value = self._calculate_portfolio_value()
        
        # Calculer les positions cibles en nombre d'actions
        target_values = target_weights * current_portfolio_value
        target_positions = target_values / current_prices
        
        # Calculer les trades (différence entre cible et actuel)
        trades = target_positions - self.portfolio_positions
        
        return trades
    
    def _execute_trades(self, trades: np.ndarray):
        """Exécute les trades et met à jour le cash et les positions"""
        current_prices = self.prices_matrix[self.current_step]
        
        for i, trade in enumerate(trades):
            if abs(trade) > 1e-6:  # Seuil minimal pour éviter les micro-trades
                trade_value = trade * current_prices[i]
                
                # Vérifier si on a assez de cash pour l'achat
                if trade > 0 and trade_value > self.cash:
                    # Ajuster le trade selon le cash disponible
                    affordable_trade = self.cash / current_prices[i]
                    trade = min(trade, affordable_trade)
                    trade_value = trade * current_prices[i]
                
                # Exécuter le trade
                self.portfolio_positions[i] += trade
                self.cash -= trade_value
                
                # Mettre à jour les poids
                portfolio_value = self._calculate_portfolio_value()
                if portfolio_value > 0:
                    self.portfolio_weights[i] = (self.portfolio_positions[i] * current_prices[i]) / portfolio_value
    
    def _update_holding_periods(self):
        """Met à jour les périodes de détention"""
        for i in range(len(self.valid_tickers)):
            if self.portfolio_weights[i] > 0.01:  # Seuil minimum
                self.holding_periods[i] += 1  # +1 semaine
            else:
                self.holding_periods[i] = 0  # Reset si position fermée
    
    def _calculate_portfolio_return(self) -> float:
        """Calcule le rendement du portefeuille"""
        if self.current_step == 0:
            return 0.0
        
        prev_value = self.portfolio_values_history[-1]
        current_value = self._calculate_portfolio_value()
        
        if prev_value > 0:
            return (current_value - prev_value) / prev_value
        return 0.0
    
    def _calculate_portfolio_value(self) -> float:
        """Calcule la valeur totale du portefeuille"""
        if self.current_step >= len(self.prices_matrix):
            return self.cash
        
        current_prices = self.prices_matrix[self.current_step]
        portfolio_value = self.cash + np.sum(self.portfolio_positions * current_prices)
        
        # Ajouter les dividendes si applicable
        if self.current_step < len(self.dividends_data):
            dividends = np.sum(self.portfolio_positions * self.dividends_data[self.current_step])
            portfolio_value += dividends
        
        return portfolio_value
    
    def _calculate_cvar(self, returns: List[float], alpha: float = None) -> float:
        """Calcule la Conditional Value at Risk (CVaR)"""
        if not returns or len(returns) < 2:
            return 0.0
        
        alpha = alpha or self.cvar_alpha
        returns_array = np.array(returns)
        
        # Calculer le VaR (quantile alpha)
        var = np.quantile(returns_array, alpha)
        
        # Calculer le CVaR (moyenne des rendements <= VaR)
        tail_returns = returns_array[returns_array <= var]
        if len(tail_returns) > 0:
            cvar = np.mean(tail_returns)
        else:
            cvar = var
        
        return cvar
    
    def _calculate_reward(self, portfolio_return: float, portfolio_value: float) -> float:
        """Calcule la reward avec CVaR selon spec.md"""
        
        # Rendement de base
        base_reward = portfolio_return
        
        # Calculer CVaR sur fenêtre glissante
        cvar_penalty = 0.0
        if len(self.returns_history) >= self.cvar_window:
            recent_returns = self.returns_history[-self.cvar_window:]
            cvar = self._calculate_cvar(recent_returns)
            
            # Pénalité CVaR (plus le CVaR est négatif, plus la pénalité est grande)
            if cvar < 0:
                cvar_penalty = abs(cvar) * self.cvar_lambda
        
        # Reward finale avec CVaR
        reward = base_reward - cvar_penalty
        
        # Pénalités pour violations de contraintes
        constraint_penalty = self._calculate_constraint_penalty()
        reward -= constraint_penalty
        
        # Bonus pour diversification
        diversification_bonus = self._calculate_diversification_bonus()
        reward += diversification_bonus
        
        return reward
    
    def _calculate_constraint_penalty(self) -> float:
        """Calcule les pénalités pour violations de contraintes"""
        penalty = 0.0
        
        # Pénalité pour trop d'assets
        num_active_positions = np.sum(self.portfolio_weights > 0.01)
        if num_active_positions > self.max_assets:
            penalty += 0.01 * (num_active_positions - self.max_assets)
        
        # Pénalité pour cash négatif (levier non autorisé)
        if self.cash < 0:
            penalty += 0.05 * abs(self.cash) / self.initial_cash
        
        return penalty
    
    def _calculate_diversification_bonus(self) -> float:
        """Calcule le bonus de diversification"""
        # Bonus basé sur l'entropie des poids (favorise la diversification)
        active_weights = self.portfolio_weights[self.portfolio_weights > 0.01]
        if len(active_weights) > 1:
            # Normaliser les poids actifs
            normalized_weights = active_weights / active_weights.sum()
            # Calculer l'entropie
            entropy = -np.sum(normalized_weights * np.log(normalized_weights + 1e-8))
            # Bonus proportionnel à l'entropie
            max_entropy = np.log(len(active_weights))
            if max_entropy > 0:
                diversity_bonus = 0.001 * (entropy / max_entropy)
                return diversity_bonus
        
        return 0.0
    
    def _get_observation(self) -> np.ndarray:
        """Construit l'observation actuelle"""
        if self.current_step >= len(self.features_matrix):
            # Dernière observation disponible
            features = self.features_matrix[-1].flatten()
        else:
            features = self.features_matrix[self.current_step].flatten()
        
        # État du portefeuille
        portfolio_state = self.portfolio_weights.copy()
        
        # Cash normalisé
        cash_ratio = np.array([self.cash / self.initial_cash])
        
        # Holding periods normalisées
        holding_periods_norm = self.holding_periods / 52  # Normaliser par 1 an
        
        # Concaténer tous les éléments
        observation = np.concatenate([
            features,
            portfolio_state,
            cash_ratio,
            holding_periods_norm
        ]).astype(np.float32)
        
        return observation
    
    def _get_info(self) -> Dict:
        """Retourne les informations supplémentaires"""
        portfolio_value = self._calculate_portfolio_value()
        
        info = {
            'step': self.current_step,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'portfolio_weights': self.portfolio_weights.copy(),
            'num_active_positions': np.sum(self.portfolio_weights > 0.01),
            'total_return': (portfolio_value - self.initial_cash) / self.initial_cash,
        }
        
        # Ajouter CVaR si historique suffisant
        if len(self.returns_history) >= self.cvar_window:
            recent_returns = self.returns_history[-self.cvar_window:]
            info['cvar'] = self._calculate_cvar(recent_returns)
        
        return info
    
    def render(self, mode='human'):
        """Affiche l'état actuel de l'environnement"""
        if mode == 'human':
            portfolio_value = self._calculate_portfolio_value()
            print(f"\n=== Step {self.current_step} ===")
            print(f"Portfolio Value: {portfolio_value:,.2f}")
            print(f"Cash: {self.cash:,.2f}")
            print(f"Total Return: {(portfolio_value - self.initial_cash) / self.initial_cash:.2%}")
            
            # Afficher les positions actives
            active_positions = [(i, ticker, weight) for i, (ticker, weight) in 
                              enumerate(zip(self.valid_tickers, self.portfolio_weights)) 
                              if weight > 0.01]
            
            if active_positions:
                print("Active Positions:")
                for i, ticker, weight in active_positions:
                    holding_weeks = self.holding_periods[i]
                    print(f"  {ticker}: {weight:.1%} (held {holding_weeks:.0f} weeks)")


def test_environment():
    """Test de l'environnement RL"""
    print("🧪 Test de l'environnement PortfolioEnv...")
    
    # Initialiser les composants
    data_handler = DataHandler()
    feature_processor = FeatureProcessor(data_handler)
    
    # Charger les données
    data_handler.load_all_data()
    
    # Obtenir les tickers valides pour la période de test
    valid_tickers = data_handler.get_available_tickers_for_period(
        "2010-01-01", "2012-12-31", min_observations=100
    )
    
    # Prendre les 8 premiers tickers pour le test
    test_tickers = valid_tickers[:8]
    
    # Créer l'environnement
    env = PortfolioEnv(
        tickers=test_tickers,
        start_date="2010-01-01",
        end_date="2012-12-31",
        data_handler=data_handler,
        feature_processor=feature_processor
    )
    
    # Test de reset
    obs, info = env.reset()
    print(f"✅ Reset OK - Observation shape: {obs.shape}")
    print(f"   Info: {info}")
    
    # Test de quelques steps avec actions aléatoires
    total_reward = 0
    for i in range(5):
        # Action aléatoire (allocation de portefeuille)
        action = np.random.random(len(test_tickers))
        action = action / action.sum() * 0.8  # 80% du capital investi
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {i+1}: Reward={reward:.4f}, Portfolio Value={info['portfolio_value']:.2f}")
        
        if terminated:
            break
    
    print(f"✅ Test environnement terminé - Reward total: {total_reward:.4f}")
    
    # Test du rendering
    env.render()


if __name__ == "__main__":
    test_environment()
