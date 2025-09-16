# Configuration pour l'agent RL de gestion de portefeuille
import numpy as np
import torch
from datetime import datetime

class Config:
    """Configuration centrale pour l'agent SAC de gestion de portefeuille"""
    
    # Paramètres des données
    DATA_PATH = "datas/"
    TICKERS_FILE = "all_datas.xlsx"
    DIVIDENDS_FILE = "dividendes.xlsx"
    NB_ACTIONS_FILE = "nb_actions.xlsx"
    SECTORS_FILE = "actions_secteurs_pays.xlsx"
    
    # Périodes de données selon spec.md
    TRAIN_START = "1998-01-01"
    TRAIN_END = "2016-12-31"
    VALIDATION_START = "2017-01-01"
    VALIDATION_END = "2021-12-31"
    TEST_START = "2022-01-01"
    TEST_END = "2024-12-31"
    
    # Contraintes de portefeuille
    INITIAL_CASH = 1000000  # 1M FCFA
    MIN_HOLDING_WEEKS = 4   # Durée minimale de détention
    BUFFER_KEEP = [11, 12]  # Garder top 11-12 assets
    REPLACE_IF = 15         # Remplacer si position > 15
    MAX_ASSETS = 12         # Portfolio de 12 assets max
    NO_SHORT_SELLING = True # Pas de vente à découvert
    NO_LEVERAGE = True      # Pas de levier
    
    # Paramètres CVaR pour gestion du risque
    CVAR_ALPHA = 0.05       # Niveau de confiance 95%
    CVAR_LAMBDA = 0.5       # Poids CVaR dans reward
    CVAR_WINDOW = 52        # Fenêtre glissante 52 semaines
    
    # Paramètres des indicateurs techniques (21 au total)
    TECHNICAL_INDICATORS = [
        'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',      # Simple Moving Averages
        'EMA_12', 'EMA_26',                          # Exponential Moving Averages
        'RSI_14',                                    # Relative Strength Index
        'MACD', 'MACD_signal', 'MACD_histogram',     # MACD components
        'BB_upper', 'BB_middle', 'BB_lower', 'BB_width', # Bollinger Bands
        'ATR_14',                                    # Average True Range
        'Volume_SMA_20',                             # Volume indicator
        'Price_change_1d', 'Price_change_5d',        # Price changes
        'Volatility_20d',                            # Volatility
        'High_Low_ratio',                            # High/Low ratio
        'Close_Volume_ratio'                         # Close/Volume ratio
    ]
    
    # Architecture du réseau de neurones
    FEATURE_DIM = len(TECHNICAL_INDICATORS)  # 21 features par asset
    HIDDEN_DIM = 256
    ATTENTION_HEADS = 8
    ATTENTION_DIM = 64
    
    # Paramètres SAC
    ACTOR_LR = 3e-4
    CRITIC_LR = 3e-4
    ALPHA_LR = 3e-4
    INITIAL_ALPHA = 0.2     # Entropy regularization
    GAMMA = 0.99            # Discount factor
    TAU = 0.005             # Soft update rate
    BATCH_SIZE = 256
    REPLAY_BUFFER_SIZE = 1000000
    
    # Paramètres d'entraînement
    MAX_EPISODES = 1000
    MAX_STEPS_PER_EPISODE = 252  # ~1 an de trading
    EVAL_FREQUENCY = 50          # Évaluer tous les 50 episodes
    SAVE_FREQUENCY = 100         # Sauvegarder tous les 100 episodes
    
    # Seeds pour reproductibilité
    RANDOM_SEED = 42
    TORCH_SEED = 42
    NUMPY_SEED = 42
    
    # Logging et sauvegarde
    LOG_DIR = "logs/"
    MODEL_DIR = "models/"
    RESULTS_DIR = "results/"
    
    # Dimensions par défaut pour l'agent
    DEFAULT_AGENT_NUM_ASSETS = 10
    DEFAULT_AGENT_STATE_DIM = 231
    DEFAULT_AGENT_ACTION_DIM = 10
    
    # Paramètres de performance - GPU par défaut avec fallback CPU
    @classmethod
    def get_device(cls):
        """Détection intelligente du device optimal (MPS > CUDA > CPU)"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    # Initialiser le device au chargement de la classe
    DEVICE = None  # Sera initialisé lors du premier appel à get_device()
    NUM_WORKERS = 4
    
    @classmethod 
    def init_device(cls):
        """Initialise le device si pas encore fait"""
        if cls.DEVICE is None:
            cls.DEVICE = cls.get_device()
        return cls.DEVICE
    
    @classmethod
    def get_data_split_dates(cls):
        """Retourne les dates de division des données"""
        return {
            'train': (cls.TRAIN_START, cls.TRAIN_END),
            'validation': (cls.VALIDATION_START, cls.VALIDATION_END),
            'test': (cls.TEST_START, cls.TEST_END)
        }
    
    @classmethod
    def validate_config(cls):
        """Validation de la configuration"""
        assert len(cls.TECHNICAL_INDICATORS) == 21, "Il faut exactement 21 indicateurs techniques"
        assert cls.CVAR_ALPHA > 0 and cls.CVAR_ALPHA < 1, "CVaR alpha doit être entre 0 et 1"
        assert cls.MAX_ASSETS >= max(cls.BUFFER_KEEP), "MAX_ASSETS doit être >= max(BUFFER_KEEP)"
        assert cls.MIN_HOLDING_WEEKS > 0, "MIN_HOLDING_WEEKS doit être positif"
        print("✅ Configuration validée avec succès")

if __name__ == "__main__":
    Config.validate_config()
    # Initialiser le device
    device = Config.init_device()
    print(f"Device détecté: {device}")
    print("Configuration chargée:")
    print(f"- Période d'entraînement: {Config.TRAIN_START} à {Config.TRAIN_END}")
    print(f"- Période de validation: {Config.VALIDATION_START} à {Config.VALIDATION_END}")
    print(f"- Période de test: {Config.TEST_START} à {Config.TEST_END}")
    print(f"- Nombre d'indicateurs techniques: {len(Config.TECHNICAL_INDICATORS)}")
    print(f"- Contraintes portefeuille: max {Config.MAX_ASSETS} assets, holding min {Config.MIN_HOLDING_WEEKS} semaines")
    print(f"- CVaR: α={Config.CVAR_ALPHA}, λ={Config.CVAR_LAMBDA}, fenêtre={Config.CVAR_WINDOW}")
