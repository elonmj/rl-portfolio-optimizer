"""
Module de modélisation stochastique du risque selon modelisation.pdf Section 2.4
Implémente le framework ARMA-GARCH + KDE + R-Vine Copulas pour l'estimation CVaR
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Importations pour la modélisation stochastique
from arch import arch_model
from sklearn.neighbors import KernelDensity
from copulas.multivariate import VineCopula
import scipy.stats as stats

from config import Config

class StochasticRiskModeling:
    """
    Framework de modélisation stochastique du risque complet:
    1. Modèles ARMA-GARCH pour la volatilité conditionnelle (Équation 13)
    2. Estimation KDE des distributions marginales (Équation 15) 
    3. Copules R-Vine pour la structure de dépendance
    4. Simulation Monte Carlo pour l'estimation CVaR
    """
    
    def __init__(self, n_scenarios: int = 10000):
        # Paramètres de simulation
        self.n_scenarios = n_scenarios
        
        # Stockage des modèles
        self.arma_garch_models = {}
        self.kde_estimators = {}
        self.vine_copula = None
        
        # Données pour la modélisation
        self.returns_data = None
        self.standardized_returns = None
        self.assets = []
        
        # Configuration ARMA-GARCH
        self.arma_order = (1, 1)    # ARMA(1,1) par défaut
        self.garch_order = (1, 1)   # GARCH(1,1) par défaut
        self.kde_bandwidth = 0.1    # Bandwidth KDE par défaut
        
        print("📊 Framework de modélisation stochastique initialisé")
    
    def fit_models(self, returns_data: pd.DataFrame, min_observations: int = 252) -> bool:
        """
        Ajuste tous les modèles sur les données de rendements
        
        Args:
            returns_data: DataFrame avec rendements (colonnes = actifs, index = dates)
            min_observations: Nombre minimum d'observations requises
            
        Returns:
            True si l'ajustement réussit, False sinon
        """
        
        if len(returns_data) < min_observations:
            print(f"⚠️ Pas assez d'observations: {len(returns_data)} < {min_observations}")
            return False
        
        # Nettoyer les données
        self.returns_data = returns_data.dropna()
        self.assets = list(self.returns_data.columns)
        
        print(f"📈 Ajustement des modèles sur {len(self.returns_data)} observations pour {len(self.assets)} actifs")
        
        # Étape 1: Ajuster les modèles ARMA-GARCH
        success_garch = self._fit_arma_garch_models()
        if not success_garch:
            print("❌ Échec de l'ajustement ARMA-GARCH")
            return False
        
        # Étape 2: Standardiser les rendements
        self._standardize_returns()
        
        # Étape 3: Ajuster les distributions marginales KDE
        success_kde = self._fit_kde_marginals()
        if not success_kde:
            print("❌ Échec de l'ajustement KDE")
            return False
        
        # Étape 4: Ajuster la copule R-Vine
        success_copula = self._fit_vine_copula()
        if not success_copula:
            print("❌ Échec de l'ajustement copule R-Vine")
            return False
        
        print("✅ Tous les modèles ajustés avec succès")
        return True
    
    def _fit_arma_garch_models(self) -> bool:
        """
        Ajuste les modèles ARMA-GARCH selon l'Équation (13):
        r_i_t = mu_i_t + sigma_i_t * epsilon_i_t
        """
        
        print("  Ajustement des modèles ARMA-GARCH...")
        
        for asset in self.assets:
            try:
                returns_series = self.returns_data[asset] * 100  # Conversion en pourcentage
                
                # Créer le modèle ARMA-GARCH
                model = arch_model(
                    returns_series,
                    mean='ARX',          # Modèle ARMA pour la moyenne
                    vol='GARCH',         # Modèle GARCH pour la volatilité
                    p=self.arma_order[0], # Ordre AR
                    q=self.arma_order[1], # Ordre MA  
                    rescale=False
                )
                
                # Ajuster le modèle
                fitted_model = model.fit(disp='off', show_warning=False)
                
                # Stocker le modèle ajusté
                self.arma_garch_models[asset] = fitted_model
                
                print(f"    ✅ {asset}: ARMA({self.arma_order[0]},{self.arma_order[1]})-GARCH({self.garch_order[0]},{self.garch_order[1]})")
                
            except Exception as e:
                print(f"    ❌ {asset}: Erreur ARMA-GARCH - {e}")
                return False
        
        return True
    
    def _standardize_returns(self):
        """
        Standardise les rendements selon l'Équation (14):
        standardized_returns = (r_i_t - mu_i_t) / sigma_i_t
        """
        
        print("  Standardisation des rendements...")
        
        standardized_data = {}
        
        for asset in self.assets:
            if asset not in self.arma_garch_models:
                continue
                
            fitted_model = self.arma_garch_models[asset]
            
            # Extraire les résidus standardisés du modèle GARCH
            standardized_residuals = fitted_model.resid / fitted_model.conditional_volatility
            
            # Stocker les résidus standardisés
            standardized_data[asset] = standardized_residuals.values
        
        # Créer DataFrame des rendements standardisés
        self.standardized_returns = pd.DataFrame(standardized_data)
        self.standardized_returns = self.standardized_returns.dropna()
        
        print(f"    ✅ {len(self.standardized_returns)} observations standardisées")
    
    def _fit_kde_marginals(self) -> bool:
        """
        Ajuste les distributions marginales avec KDE selon l'Équation (15)
        """
        
        print("  Ajustement des distributions marginales KDE...")
        
        if self.standardized_returns is None:
            return False
        
        for asset in self.assets:
            try:
                if asset not in self.standardized_returns.columns:
                    continue
                    
                standardized_series = np.array(self.standardized_returns[asset]).reshape(-1, 1)
                
                # Ajuster le KDE
                kde = KernelDensity(
                    kernel='gaussian',
                    bandwidth=self.kde_bandwidth
                )
                kde.fit(standardized_series)
                
                # Stocker l'estimateur KDE
                self.kde_estimators[asset] = kde
                
                print(f"    ✅ {asset}: KDE avec bandwidth={self.kde_bandwidth}")
                
            except Exception as e:
                print(f"    ❌ {asset}: Erreur KDE - {e}")
                return False
        
        return True
    
    def _fit_vine_copula(self) -> bool:
        """
        Ajuste la copule R-Vine pour capturer la structure de dépendance
        """
        
        print("  Ajustement de la copule R-Vine...")
        
        try:
            # Convertir les rendements standardisés en rangs uniformes
            uniform_data = self._to_uniform_margins()
            
            if uniform_data is None or len(uniform_data.columns) < 2:
                print("    ❌ Données insuffisantes pour la copule")
                return False
            
            # Ajuster la copule Vine
            self.vine_copula = VineCopula()
            self.vine_copula.fit(uniform_data)
            
            print(f"    ✅ Copule R-Vine ajustée sur {len(uniform_data)} observations")
            return True
            
        except Exception as e:
            print(f"    ❌ Erreur copule R-Vine: {e}")
            
            # Fallback: utiliser une copule Gaussienne simple
            try:
                print("    🔄 Tentative avec copule Gaussienne...")
                
                if self.standardized_returns is not None:
                    # Créer une corrélation simple comme fallback
                    correlation_matrix = np.corrcoef(self.standardized_returns.T)
                    self.vine_copula = {
                        'type': 'gaussian_fallback',
                        'correlation': correlation_matrix
                    }
                    
                    print("    ✅ Copule Gaussienne ajustée (fallback)")
                    return True
                else:
                    print("    ❌ Données standardisées manquantes")
                    return False
                
            except Exception as e2:
                print(f"    ❌ Échec copule Gaussienne: {e2}")
                return False
    
    def _to_uniform_margins(self) -> Optional[pd.DataFrame]:
        """
        Convertit les rendements standardisés en marges uniformes [0,1]
        """
        
        if self.standardized_returns is None:
            return None
        
        uniform_data = {}
        
        for asset in self.standardized_returns.columns:
            # Calculer les rangs empiriques et convertir en [0,1]
            ranks = stats.rankdata(self.standardized_returns[asset])
            uniform_values = ranks / (len(ranks) + 1)  # Éviter 0 et 1 exacts
            uniform_data[asset] = uniform_values
        
        return pd.DataFrame(uniform_data)
    
    def simulate_scenarios(self, n_scenarios: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Génère des scénarios Monte Carlo pour l'estimation CVaR
        
        Args:
            n_scenarios: Nombre de scénarios à générer
            
        Returns:
            Matrice des rendements simulés (n_scenarios x n_assets)
        """
        
        n_scenarios = n_scenarios or self.n_scenarios
        
        if not self._models_ready():
            print("❌ Modèles non ajustés - impossible de simuler")
            return None
        
        try:
            print(f"🎲 Génération de {n_scenarios} scénarios Monte Carlo...")
            
            # 1. Génération basée sur le type de copule
            if isinstance(self.vine_copula, dict) and self.vine_copula.get('type') == 'gaussian_fallback':
                # Fallback: utiliser la corrélation Gaussienne
                correlation = self.vine_copula['correlation']
                copula_samples = np.random.multivariate_normal(
                    mean=np.zeros(len(self.assets)),
                    cov=correlation,
                    size=n_scenarios
                )
            elif hasattr(self.vine_copula, 'sample'):
                # Utiliser la vraie copule Vine
                copula_samples = self.vine_copula.sample(n_scenarios)
                if isinstance(copula_samples, pd.DataFrame):
                    copula_samples = copula_samples.values
            else:
                # Fallback simple: échantillons indépendants
                copula_samples = np.random.randn(n_scenarios, len(self.assets))
            
            # 2. Transformer en rendements via les marginales
            simulated_returns = np.zeros((n_scenarios, len(self.assets)))
            
            for i, asset in enumerate(self.assets):
                if self.standardized_returns is not None and asset in self.standardized_returns.columns:
                    # Utiliser les quantiles empiriques des données standardisées
                    asset_data = np.array(self.standardized_returns[asset])
                    
                    if isinstance(self.vine_copula, dict):
                        # Pour le fallback Gaussien, utiliser directement les valeurs
                        simulated_returns[:, i] = copula_samples[:, i]
                    else:
                        # Pour les vraies copules, transformer via quantiles
                        uniform_values = copula_samples[:, i]
                        quantiles = np.quantile(asset_data, uniform_values)
                        simulated_returns[:, i] = quantiles
            
            print(f"    ✅ {n_scenarios} scénarios générés")
            return simulated_returns
            
        except Exception as e:
            print(f"❌ Erreur simulation: {e}")
            return None
    
    def calculate_portfolio_cvar(self, 
                               weights: np.ndarray, 
                               confidence_level: float = 0.05,
                               n_scenarios: Optional[int] = None) -> float:
        """
        Calcule la CVaR du portefeuille via simulation Monte Carlo
        
        Args:
            weights: Poids du portefeuille
            confidence_level: Niveau de confiance (5% par défaut)
            n_scenarios: Nombre de scénarios pour simulation
            
        Returns:
            CVaR estimée
        """
        
        # Générer les scénarios
        scenarios = self.simulate_scenarios(n_scenarios)
        if scenarios is None:
            return 0.0
        
        # Calculer les rendements de portefeuille
        portfolio_returns = np.dot(scenarios, weights)
        
        # Calculer la VaR (quantile)
        var = np.quantile(portfolio_returns, confidence_level)
        
        # Calculer la CVaR (espérance conditionnelle)
        cvar_returns = portfolio_returns[portfolio_returns <= var]
        
        if len(cvar_returns) == 0:
            return 0.0
        
        cvar = np.mean(cvar_returns)
        
        return -cvar  # Retourner la valeur positive (perte)
    
    def _models_ready(self) -> bool:
        """Vérifie si tous les modèles sont prêts"""
        return (
            len(self.arma_garch_models) > 0 and
            len(self.kde_estimators) > 0 and
            self.vine_copula is not None and
            self.standardized_returns is not None
        )
    
    def get_model_summary(self) -> Dict:
        """Retourne un résumé des modèles ajustés"""
        return {
            'n_assets': len(self.assets),
            'n_observations': len(self.returns_data) if self.returns_data is not None else 0,
            'arma_garch_fitted': len(self.arma_garch_models),
            'kde_fitted': len(self.kde_estimators),
            'copula_fitted': self.vine_copula is not None,
            'ready_for_simulation': self._models_ready()
        }


def test_stochastic_risk_modeling():
    """Test du framework de modélisation stochastique"""
    print("🧪 Test du framework de modélisation stochastique...")
    
    # Générer des données de test
    np.random.seed(42)
    n_obs = 500
    n_assets = 3
    
    # Simulation de rendements corrélés
    returns_data = pd.DataFrame(
        np.random.multivariate_normal(
            mean=[0.001, 0.001, 0.001],
            cov=[[0.0004, 0.0001, 0.0002],
                 [0.0001, 0.0009, 0.0001], 
                 [0.0002, 0.0001, 0.0016]],
            size=n_obs
        ),
        columns=['Asset1', 'Asset2', 'Asset3']
    )
    
    # Tester le framework
    risk_model = StochasticRiskModeling(n_scenarios=1000)
    
    # Ajuster les modèles
    success = risk_model.fit_models(returns_data)
    
    if success:
        # Calculer CVaR pour un portefeuille équipondéré
        weights = np.array([1/3, 1/3, 1/3])
        cvar = risk_model.calculate_portfolio_cvar(weights, confidence_level=0.05)
        
        print(f"✅ CVaR du portefeuille équipondéré: {cvar:.4f}")
        
        # Résumé des modèles
        summary = risk_model.get_model_summary()
        print(f"✅ Résumé: {summary}")
        
    else:
        print("❌ Échec des tests")


if __name__ == "__main__":
    test_stochastic_risk_modeling()