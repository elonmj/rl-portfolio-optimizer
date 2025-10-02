"""
Module de traitement des données pour l'agent SAC de gestion de portefeuille.
Implémente DataHandler et FeatureProcessor selon spec.md.
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import Config

class DataHandler:
    """Gestionnaire principal des données financières"""
    
    def __init__(self, data_path: str = None):
        self.data_path = Path(data_path or Config.DATA_PATH)
        self.tickers_data = {}
        self.dividends_data = None
        self.nb_actions_data = None
        self.sectors_data = None
        self.available_tickers = []
        
    def load_all_data(self) -> Dict:
        """Charge toutes les données nécessaires"""
        print("📊 Chargement des données...")
        
        # Charger les données de prix (all_datas.xlsx)
        self._load_price_data()
        
        # Charger les données auxiliaires
        self._load_auxiliary_data()
        
        print(f"  Données chargées: {len(self.available_tickers)} tickers disponibles")
        return {
            'prices': self.tickers_data,
            'dividends': self.dividends_data,
            'nb_actions': self.nb_actions_data,
            'sectors': self.sectors_data,
            'tickers': self.available_tickers
        }
    
    def _load_price_data(self):
        """Charge les données de prix depuis all_datas.xlsx"""
        file_path = self.data_path / Config.TICKERS_FILE
        
        try:
            # Lire toutes les feuilles Excel
            excel_file = pd.ExcelFile(file_path)
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Nettoyer les données (supprimer NaN)
                df_clean = df.dropna()
                
                if len(df_clean) > 0:
                    # Convertir la colonne Date
                    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
                    df_clean = df_clean.sort_values('Date').reset_index(drop=True)
                    
                    # Stocker les données
                    self.tickers_data[sheet_name] = df_clean
                    self.available_tickers.append(sheet_name)
                    
        except Exception as e:
            print(f"❌ Erreur lors du chargement des prix: {e}")
            raise
    
    def _load_auxiliary_data(self):
        """Charge les données auxiliaires (dividendes, nb_actions, secteurs)"""
        try:
            # Dividendes
            div_path = self.data_path / Config.DIVIDENDS_FILE
            self.dividends_data = pd.read_excel(div_path)
            self.dividends_data = self.dividends_data.rename(columns={'Date': 'Year'})
            
            # Nombre d'actions
            nb_path = self.data_path / Config.NB_ACTIONS_FILE
            self.nb_actions_data = pd.read_excel(nb_path)
            
            # Secteurs et pays
            sect_path = self.data_path / Config.SECTORS_FILE
            self.sectors_data = pd.read_excel(sect_path)
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement des données auxiliaires: {e}")
            raise
    
    def get_ticker_data(self, ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Récupère les données d'un ticker pour une période donnée"""
        if ticker not in self.tickers_data:
            raise ValueError(f"Ticker {ticker} non disponible")
        
        df = self.tickers_data[ticker].copy()
        
        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['Date'] <= pd.to_datetime(end_date)]
        
        return df
    
    def get_available_tickers_for_period(self, start_date: str, end_date: str, min_observations: int = 100) -> List[str]:
        """Retourne les tickers ayant suffisamment de données pour la période"""
        valid_tickers = []
        
        for ticker in self.available_tickers:
            df = self.get_ticker_data(ticker, start_date, end_date)
            if len(df) >= min_observations:
                valid_tickers.append(ticker)
        
        return valid_tickers


class FeatureProcessor:
    """Processeur de features avec calcul des 21 indicateurs techniques"""
    
    def __init__(self, data_handler: DataHandler):
        self.data_handler = data_handler
        self.technical_indicators = Config.TECHNICAL_INDICATORS
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les 21 indicateurs techniques selon spec.md"""
        result_df = df.copy()
        
        # Vérifier que les colonnes nécessaires existent
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Colonnes manquantes. Requis: {required_cols}")
        
        try:
            # Simple Moving Averages
            result_df['SMA_5'] = ta.trend.sma_indicator(df['Close'], window=5)
            result_df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
            result_df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            result_df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            
            # Exponential Moving Averages
            result_df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
            result_df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
            
            # RSI
            result_df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            result_df['MACD'] = macd.macd()
            result_df['MACD_signal'] = macd.macd_signal()
            result_df['MACD_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['Close'])
            result_df['BB_upper'] = bb.bollinger_hband()
            result_df['BB_middle'] = bb.bollinger_mavg()
            result_df['BB_lower'] = bb.bollinger_lband()
            result_df['BB_width'] = bb.bollinger_wband()
            
            # Average True Range
            result_df['ATR_14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
            
            # Volume indicator
            result_df['Volume_SMA_20'] = ta.trend.sma_indicator(df['Volume'], window=20)
            
            # Price changes
            result_df['Price_change_1d'] = df['Close'].pct_change(1)
            result_df['Price_change_5d'] = df['Close'].pct_change(5)
            
            # Volatility
            result_df['Volatility_20d'] = df['Close'].rolling(window=20).std()
            
            # High/Low ratio
            result_df['High_Low_ratio'] = df['High'] / df['Low']
            
            # Close/Volume ratio
            result_df['Close_Volume_ratio'] = df['Close'] / (df['Volume'] + 1e-8)  # Éviter division par 0
            
            # Normaliser les ratios qui peuvent être très grands
            result_df['Close_Volume_ratio'] = np.log1p(result_df['Close_Volume_ratio'])
            
        except Exception as e:
            print(f"❌ Erreur lors du calcul des indicateurs pour {df.iloc[0] if len(df) > 0 else 'données vides'}: {e}")
            raise
        
        return result_df
    
    def normalize_features(self, df: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """Normalise les features techniques"""
        normalized_df = df.copy()
        
        for indicator in self.technical_indicators:
            if indicator in df.columns:
                if method == 'zscore':
                    mean = df[indicator].mean()
                    std = df[indicator].std()
                    if std > 0:
                        normalized_df[indicator] = (df[indicator] - mean) / std
                    else:
                        normalized_df[indicator] = 0
                        
                elif method == 'minmax':
                    min_val = df[indicator].min()
                    max_val = df[indicator].max()
                    if max_val > min_val:
                        normalized_df[indicator] = (df[indicator] - min_val) / (max_val - min_val)
                    else:
                        normalized_df[indicator] = 0
        
        return normalized_df
    
    def calculate_top_k_scores(self, features_dict: Dict[str, pd.DataFrame], k: int = 12) -> Dict[str, np.ndarray]:
        """Calcule les scores Top-K pour sélection d'assets selon spec.md"""
        
        # Créer une matrice de scores pour tous les tickers
        all_scores = {}
        
        for ticker, df in features_dict.items():
            if len(df) == 0:
                continue
                
            # Calculer un score composite basé sur plusieurs indicateurs
            scores = []
            
            # Score basé sur momentum (RSI, MACD)
            if 'RSI_14' in df.columns and not df['RSI_14'].isna().all():
                rsi_score = (50 - abs(df['RSI_14'] - 50)) / 50  # Favorise RSI proche de 50
                scores.append(rsi_score)
            
            # Score basé sur trend (SMA)
            if all(col in df.columns for col in ['Close', 'SMA_20']):
                trend_score = np.where(df['Close'] > df['SMA_20'], 1, -1)
                scores.append(trend_score)
            
            # Score basé sur volatilité
            if 'Volatility_20d' in df.columns and not df['Volatility_20d'].isna().all():
                vol_score = 1 / (1 + df['Volatility_20d'])  # Inverse de la volatilité
                scores.append(vol_score)
            
            # Score basé sur MACD
            if 'MACD' in df.columns and 'MACD_signal' in df.columns:
                macd_score = np.where(df['MACD'] > df['MACD_signal'], 1, -1)
                scores.append(macd_score)
            
            if scores:
                # Score composite moyen
                composite_score = np.mean(scores, axis=0)
                # Lisser avec moyenne mobile
                if len(composite_score) > 5:
                    composite_score = pd.Series(composite_score).rolling(window=5, min_periods=1).mean().values
                all_scores[ticker] = composite_score
        
        return all_scores
    
    def prepare_features_matrix(self, tickers: List[str], start_date: str, end_date: str) -> Tuple[np.ndarray, List[str], pd.DatetimeIndex]:
        """Prépare la matrice de features pour l'environnement RL"""
        
        features_data = {}
        common_dates = None
        
        # Collecter les données et calculer les indicateurs
        for ticker in tickers:
            try:
                df = self.data_handler.get_ticker_data(ticker, start_date, end_date)
                if len(df) < 50:  # Minimum de données requis
                    continue
                
                # Calculer les indicateurs techniques
                df_with_indicators = self.calculate_technical_indicators(df)
                
                # Normaliser
                df_normalized = self.normalize_features(df_with_indicators)
                
                # Supprimer les NaN
                df_clean = df_normalized.dropna()
                
                if len(df_clean) > 0:
                    features_data[ticker] = df_clean
                    
                    # Trouver les dates communes
                    if common_dates is None:
                        common_dates = set(df_clean['Date'])
                    else:
                        common_dates = common_dates.intersection(set(df_clean['Date']))
                        
            except Exception as e:
                print(f"⚠️ Erreur avec ticker {ticker}: {e}")
                continue
        
        # Aligner toutes les données sur les dates communes
        if not common_dates:
            raise ValueError("Aucune date commune trouvée entre les tickers")
        
        common_dates = sorted(list(common_dates))
        aligned_features = []
        valid_tickers = []
        
        for ticker in tickers:
            if ticker in features_data:
                df = features_data[ticker]
                df_aligned = df[df['Date'].isin(common_dates)].sort_values('Date')
                
                if len(df_aligned) == len(common_dates):
                    # Extraire seulement les indicateurs techniques
                    feature_matrix = df_aligned[self.technical_indicators].values
                    aligned_features.append(feature_matrix)
                    valid_tickers.append(ticker)
        
        if not aligned_features:
            raise ValueError("Aucune donnée alignée disponible")
        
        # Créer la matrice finale (T, K, 21) où T=temps, K=assets, 21=features
        features_matrix = np.stack(aligned_features, axis=1)  # Shape: (T, K, 21)
        
        print(f"  Matrice de features créée: {features_matrix.shape} (temps, assets, features)")
        
        return features_matrix, valid_tickers, pd.DatetimeIndex(common_dates)


def test_data_processing():
    """Test du module de traitement des données"""
    print("🧪 Test du module data_processing...")
    
    # Initialiser
    data_handler = DataHandler()
    feature_processor = FeatureProcessor(data_handler)
    
    # Charger les données
    all_data = data_handler.load_all_data()
    
    # Tester sur la période d'entraînement
    valid_tickers = data_handler.get_available_tickers_for_period(
        Config.TRAIN_START, 
        Config.TRAIN_END,
        min_observations=200
    )
    
    print(f"Tickers valides pour entraînement: {len(valid_tickers)}")
    
    if len(valid_tickers) > 0:
        # Prendre les 5 premiers pour le test
        test_tickers = valid_tickers[:5]
        
        # Créer la matrice de features
        features_matrix, final_tickers, dates = feature_processor.prepare_features_matrix(
            test_tickers, 
            "2010-01-01", 
            "2015-12-31"
        )
        
        print(f"Test réussi! Matrice: {features_matrix.shape}")
        print(f"Tickers finaux: {final_tickers}")
        print(f"Période: {dates[0]} à {dates[-1]}")
        
        # Vérifier qu'il n'y a pas de NaN
        if np.isnan(features_matrix).any():
            print("⚠️ Attention: des NaN détectés dans la matrice")
        else:
            print("  Aucun NaN dans la matrice de features")
    
    print("  Test terminé avec succès!")


class StockPickingProcessor:
    """
    Module de sélection d'actions avec critères multiples selon modelisation.pdf Section 2.1
    Implémente l'algorithme de sélection des top-K actifs basé sur:
    - Momentum (performance relative)
    - Volatilité (risque)
    - Liquidité (volume moyen)
    - Dividendes (rendement)
    """
    
    def __init__(self, data_handler: DataHandler):
        self.data_handler = data_handler
        
        # Poids des critères selon modelisation.pdf
        self.w_momentum = 0.3      # Poids momentum
        self.w_volatility = 0.25   # Poids volatilité (négatif dans le score)
        self.w_liquidity = 0.25    # Poids liquidité
        self.w_dividend = 0.2      # Poids dividendes
        
        # Paramètres de la fenêtre de calcul
        self.window_W = 52         # Fenêtre de 52 semaines (1 an)
        
    def select_top_k_assets(self, 
                          universe: List[str], 
                          date_t: str, 
                          k: int = 10,
                          min_history: int = 52) -> List[str]:
        """
        Sélectionne les top-K actifs selon l'algorithme de modelisation.pdf
        
        Args:
            universe: Liste des tickers disponibles
            date_t: Date de sélection
            k: Nombre d'actifs à sélectionner
            min_history: Nombre minimum d'observations requises
            
        Returns:
            Liste des K meilleurs actifs sélectionnés
        """
        
        date_end = pd.to_datetime(date_t)
        date_start = date_end - pd.Timedelta(weeks=self.window_W + 4)  # Marge de sécurité
        
        scores = {}
        
        for ticker in universe:
            try:
                # Récupérer les données historiques
                df = self.data_handler.get_ticker_data(
                    ticker, 
                    date_start.strftime('%Y-%m-%d'), 
                    date_end.strftime('%Y-%m-%d')
                )
                
                if len(df) < min_history:
                    continue
                    
                # Calculer les métriques selon modelisation.pdf
                momentum = self._calculate_momentum(df)
                volatility = self._calculate_volatility(df)
                liquidity = self._calculate_liquidity(df)
                dividend_yield = self._calculate_dividend_yield(ticker, date_t)
                
                # Vérifier que toutes les métriques sont valides
                if all(not np.isnan(x) for x in [momentum, volatility, liquidity, dividend_yield]):
                    scores[ticker] = {
                        'momentum': momentum,
                        'volatility': volatility,
                        'liquidity': liquidity,
                        'dividend': dividend_yield
                    }
                    
            except Exception as e:
                print(f"⚠️ Erreur pour {ticker}: {e}")
                continue
        
        if len(scores) < k:
            print(f"⚠️ Seulement {len(scores)} actifs valides trouvés pour {k} demandés")
            k = len(scores)
        
        # Calculer les rangs et le score composite
        ranked_scores = self._calculate_composite_scores(scores)
        
        # Sélectionner les top-K
        top_k = sorted(ranked_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        selected_assets = [asset for asset, score in top_k]
        
        print(f"📊 Sélection d'actifs pour {date_t}:")
        print(f"  Top-{k} actifs: {selected_assets}")
        
        return selected_assets
    
    def _calculate_momentum(self, df: pd.DataFrame) -> float:
        """
        Calcule le momentum selon l'Équation: momentum_i = P[i][t-1] / P[i][t-W-1]
        """
        if len(df) < self.window_W + 2:
            return np.nan
            
        # Prix à t-1 (dernière observation)
        price_t_minus_1 = df['Close'].iloc[-1]
        
        # Prix à t-W-1 (W semaines avant)
        if len(df) >= self.window_W + 1:
            price_t_minus_W_minus_1 = df['Close'].iloc[-(self.window_W + 1)]
        else:
            price_t_minus_W_minus_1 = df['Close'].iloc[0]
        
        if price_t_minus_W_minus_1 <= 0:
            return np.nan
            
        momentum = price_t_minus_1 / price_t_minus_W_minus_1
        return momentum
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """
        Calcule la volatilité comme écart-type des rendements quotidiens
        """
        if len(df) < 2:
            return np.nan
            
        # Calculer les rendements quotidiens
        returns = df['Close'].pct_change().dropna()
        
        if len(returns) < self.window_W // 4:  # Au moins ~13 observations
            return np.nan
            
        # Volatilité annualisée (assume 252 jours de trading par an)
        daily_volatility = returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        return annualized_volatility
    
    def _calculate_liquidity(self, df: pd.DataFrame) -> float:
        """
        Calcule la liquidité comme volume moyen sur la fenêtre
        """
        if len(df) < self.window_W // 4:
            return np.nan
            
        # Volume moyen sur la période
        mean_volume = df['Volume'].mean()
        
        # Normaliser par la médiane pour éviter les valeurs extrêmes
        median_volume = df['Volume'].median()
        if median_volume > 0:
            normalized_liquidity = mean_volume / median_volume
        else:
            normalized_liquidity = mean_volume
            
        return normalized_liquidity
    
    def _calculate_dividend_yield(self, ticker: str, date_t: str) -> float:
        """
        Calcule le rendement en dividendes: dividend[i][t-1] / P[i][t-1]
        """
        try:
            if self.data_handler.dividends_data is None:
                return 0.0
                
            # Récupérer le dividende le plus récent
            if ticker in self.data_handler.dividends_data.columns:
                dividend_series = pd.to_numeric(
                    self.data_handler.dividends_data[ticker], 
                    errors='coerce'
                )
                
                # Prendre le dividende moyen des dernières années
                recent_dividend = dividend_series.mean()
                if np.isnan(recent_dividend):
                    return 0.0
                    
                # Récupérer le prix actuel
                df = self.data_handler.get_ticker_data(ticker)
                if len(df) == 0:
                    return 0.0
                    
                current_price = df['Close'].iloc[-1]
                if current_price <= 0:
                    return 0.0
                    
                dividend_yield = recent_dividend / current_price
                return max(0.0, dividend_yield)  # Assurer que le yield est positif
                
        except Exception as e:
            print(f"⚠️ Erreur dividende pour {ticker}: {e}")
            
        return 0.0
    
    def _calculate_composite_scores(self, scores: Dict) -> Dict[str, float]:
        """
        Calcule le score composite selon modelisation.pdf:
        score_i = w_mu * Rank(momentum_i) - w_sigma * Rank(volatility_i) + 
                  w_L * Rank(liquidity_i) + w_D * Rank(dividend_i)
        """
        
        if not scores:
            return {}
        
        # Extraire les métriques pour chaque actif
        tickers = list(scores.keys())
        momentums = [scores[t]['momentum'] for t in tickers]
        volatilities = [scores[t]['volatility'] for t in tickers]
        liquidities = [scores[t]['liquidity'] for t in tickers]
        dividends = [scores[t]['dividend'] for t in tickers]
        
        # Calculer les rangs (1 = meilleur, len(tickers) = pire)
        # Pour momentum, liquidité, dividendes: plus élevé = meilleur rang
        # Pour volatilité: plus faible = meilleur rang
        
        def calculate_ranks(values, ascending=False):
            """Calcule les rangs (1 = meilleur)"""
            indexed_values = [(i, v) for i, v in enumerate(values)]
            indexed_values.sort(key=lambda x: x[1], reverse=not ascending)
            
            ranks = [0] * len(values)
            for rank, (original_index, _) in enumerate(indexed_values):
                ranks[original_index] = rank + 1  # Rang commence à 1
            
            return ranks
        
        momentum_ranks = calculate_ranks(momentums, ascending=False)  # Plus haut = meilleur
        volatility_ranks = calculate_ranks(volatilities, ascending=True)   # Plus bas = meilleur  
        liquidity_ranks = calculate_ranks(liquidities, ascending=False)  # Plus haut = meilleur
        dividend_ranks = calculate_ranks(dividends, ascending=False)    # Plus haut = meilleur
        
        # Calculer le score composite
        composite_scores = {}
        for i, ticker in enumerate(tickers):
            score = (self.w_momentum * momentum_ranks[i] -
                    self.w_volatility * volatility_ranks[i] +  # Soustraction car volatilité élevée = mauvais
                    self.w_liquidity * liquidity_ranks[i] +
                    self.w_dividend * dividend_ranks[i])
            
            composite_scores[ticker] = score
            
        return composite_scores


if __name__ == "__main__":
    test_data_processing()
