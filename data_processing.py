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
        
        print(f"✅ Données chargées: {len(self.available_tickers)} tickers disponibles")
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
        
        print(f"✅ Matrice de features créée: {features_matrix.shape} (temps, assets, features)")
        
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
            print("✅ Aucun NaN dans la matrice de features")
    
    print("✅ Test terminé avec succès!")


if __name__ == "__main__":
    test_data_processing()
