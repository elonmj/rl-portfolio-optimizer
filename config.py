# Configuration pour l'agent RL de gestion de portefeuille
import numpy as np
import torch
import os
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not available. Make sure environment variables are set manually.")

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
    
    # === Paramètres Stock Picking (modelisation.pdf Section 2.1) ===
    STOCK_PICKING_ENABLED = True       # Activer le module de sélection d'actions
    STOCK_PICKING_WINDOW_W = 52        # Fenêtre d'analyse (52 semaines)
    STOCK_PICKING_TOP_K = 10           # Nombre d'actifs à sélectionner
    
    # Poids des critères de sélection
    STOCK_PICKING_W_MOMENTUM = 0.3     # Poids momentum (performance relative)
    STOCK_PICKING_W_VOLATILITY = 0.25  # Poids volatilité (risque - négatif)
    STOCK_PICKING_W_LIQUIDITY = 0.25   # Poids liquidité (volume)
    STOCK_PICKING_W_DIVIDEND = 0.2     # Poids dividendes (rendement)
    
    # === Paramètres de modélisation stochastique du risque ===
    STOCHASTIC_RISK_ENABLED = True
    N_SIMULATION_SCENARIOS = 5000      # Nombre de scénarios Monte Carlo
    RISK_MODEL_WINDOW = 252            # Fenêtre pour ajustement des modèles (1 an)
    RISK_MODEL_UPDATE_FREQUENCY = 20   # Réajuster les modèles tous les N pas
    
    # === Paramètres de fonction de récompense multi-composants ===
    # Poids des composants de récompense (Équations 9-12)
    REWARD_ALPHA_CVAR = 1.0           # Poids pénalité CVaR
    REWARD_ALPHA_DRAWDOWN = 0.5       # Poids pénalité drawdown
    REWARD_ALPHA_ENTROPY = 0.1        # Poids bonus entropie
    
    # Paramètres supplémentaires
    DRAWDOWN_WINDOW = 50              # Fenêtre pour calcul du drawdown
    
    # === Paramètres de rééquilibrage avec coûts de transaction ===
    # Coûts de transaction (Équations 7-8)
    TRANSACTION_COST_ENABLED = True   # Activer les coûts de transaction
    LAMBDA_TX = 0.001                 # Coefficient frais de transaction (0.1%)
    LAMBDA_SLIP = 0.0005              # Coefficient slippage (0.05%)
    
    # Mécaniques de rééquilibrage
    USE_INTEGER_SHARES = True         # Utiliser conversion en nombre entier d'actions
    MIN_TRADE_SIZE = 100.0            # Taille minimum de trade (éviter micro-trades)
    
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
    USE_ATTENTION = False  # Temporairement activé pour comparaison
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
    MAX_EPISODES = 1             # Test rapide avec 1 episode seulement
    MAX_STEPS_PER_EPISODE = 252  # ~1 an de trading
    EVAL_FREQUENCY = 1           # Évaluer tous les 1 episodes
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
        
    # === Kaggle Integration Configuration ===
    
    @staticmethod
    def is_kaggle_environment() -> bool:
        """
        Detect if code is running in Kaggle kernel environment.
        
        Returns:
            bool: True if running in Kaggle kernel, False otherwise
        """
        kaggle_indicators = [
            os.path.exists('/kaggle/input'),
            os.path.exists('/kaggle/working'),
            'KAGGLE_KERNEL_RUN_TYPE' in os.environ,
            'KAGGLE_URL_BASE' in os.environ
        ]
        return any(kaggle_indicators)
    
    @classmethod
    def get_execution_mode(cls) -> str:
        """
        Determine current execution mode.
        
        Returns:
            'kaggle' if in Kaggle environment, 'local' otherwise
        """
        return 'kaggle' if cls.is_kaggle_environment() else 'local'
    
    @classmethod
    def get_data_paths(cls) -> dict:
        """
        Get appropriate data paths for current execution environment.
        
        Returns:
            Dictionary with environment-specific paths
        """
        if cls.is_kaggle_environment():
            return {
                'data': '/kaggle/input',
                'working': '/kaggle/working',
                'output': '/kaggle/working',
                'models': '/kaggle/working/models',
                'results': '/kaggle/working/results',
                'logs': '/kaggle/working/logs'
            }
        else:
            # Local development paths
            return {
                'data': cls.DATA_PATH,
                'working': str(Path.cwd()),
                'output': cls.RESULTS_DIR,
                'models': cls.MODEL_DIR,
                'results': cls.RESULTS_DIR,
                'logs': cls.LOG_DIR
            }
    
    @classmethod
    def get_kaggle_config(cls) -> dict:
        """
        Get Kaggle-specific configuration from environment variables.
        
        Returns:
            Dictionary with Kaggle configuration settings
        """
        return {
            'kernel_name': os.environ.get('KAGGLE_KERNEL_NAME', 'rl-portfolio-optimizer'),
            'enable_gpu': os.environ.get('KAGGLE_ENABLE_GPU', 'true').lower() == 'true',
            'enable_tpu': os.environ.get('KAGGLE_ENABLE_TPU', 'false').lower() == 'true',
            'enable_internet': os.environ.get('KAGGLE_ENABLE_INTERNET', 'true').lower() == 'true',
            'is_private': os.environ.get('KAGGLE_KERNEL_PRIVATE', 'true').lower() == 'true',
            'auto_upload': os.environ.get('KAGGLE_AUTO_UPLOAD', 'true').lower() == 'true',
            'auto_monitor': os.environ.get('KAGGLE_AUTO_MONITOR', 'true').lower() == 'true',
            'auto_download': os.environ.get('KAGGLE_AUTO_DOWNLOAD', 'true').lower() == 'true',
            'monitor_interval': int(os.environ.get('KAGGLE_MONITOR_INTERVAL', '30')),
            'keywords': os.environ.get('KAGGLE_KERNEL_KEYWORDS', 
                                     'reinforcement-learning,portfolio,sac,pytorch,gpu').split(','),
            'description': os.environ.get('KAGGLE_KERNEL_DESCRIPTION', 
                                        'RL Portfolio Optimization with SAC Agent and GPU Acceleration')
        }
    
    @classmethod
    def get_device_for_kaggle(cls) -> torch.device:
        """
        Get appropriate device configuration for Kaggle environment.
        Prioritizes CUDA in Kaggle environment, falls back to existing logic locally.
        
        Returns:
            PyTorch device object
        """
        if cls.is_kaggle_environment():
            # In Kaggle environment, prioritize CUDA
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            # Use existing device detection for local development
            return cls.get_device()
    
    @classmethod
    def get_training_config_for_environment(cls) -> dict:
        """
        Get training configuration adapted for current environment.
        
        Returns:
            Dictionary with environment-optimized training parameters
        """
        base_config = {
            'max_episodes': cls.MAX_EPISODES,
            'batch_size': cls.BATCH_SIZE,
            'eval_frequency': cls.EVAL_FREQUENCY,
            'save_frequency': cls.SAVE_FREQUENCY
        }
        
        if cls.is_kaggle_environment():
            # Kaggle environment optimizations
            kaggle_config = {
                'max_episodes': int(os.environ.get('TRAINING_EPISODES', cls.MAX_EPISODES)),
                'batch_size': int(os.environ.get('TRAINING_BATCH_SIZE', cls.BATCH_SIZE)),
                'learning_rate': float(os.environ.get('TRAINING_LEARNING_RATE', cls.ACTOR_LR)),
                'save_interval': int(os.environ.get('TRAINING_SAVE_INTERVAL', cls.SAVE_FREQUENCY)),
                'use_cuda_optimization': True,
                'memory_efficient': True
            }
            base_config.update(kaggle_config)
        
        return base_config
    
    @classmethod
    def setup_environment_paths(cls) -> None:
        """
        Setup directory structure for current environment.
        Creates necessary directories if they don't exist.
        """
        paths = cls.get_data_paths()
        
        # Create output directories
        for dir_type in ['models', 'results', 'logs']:
            dir_path = Path(paths[dir_type])
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"  Environment paths setup completed for {cls.get_execution_mode()} mode")
    
    @classmethod
    def log_environment_info(cls) -> None:
        """Log current environment configuration and status."""
        mode = cls.get_execution_mode()
        device = cls.get_device_for_kaggle()
        paths = cls.get_data_paths()
        
        print(f"🏃 Execution Mode: {mode.upper()}")
        print(f"🖥️  Device: {device}")
        print(f"📁 Data Path: {paths['data']}")
        print(f"📁 Output Path: {paths['output']}")
        
        if mode == 'kaggle':
            kaggle_config = cls.get_kaggle_config()
            print(f"🚀 Kaggle GPU Enabled: {kaggle_config['enable_gpu']}")
            print(f"🌐 Kaggle Internet: {kaggle_config['enable_internet']}")
            print(f"🔒 Kernel Private: {kaggle_config['is_private']}")
        
        print("=" * 50)
    
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
        print("  Configuration validée avec succès")

if __name__ == "__main__":
    Config.validate_config()
    # Setup environment and log info
    Config.setup_environment_paths()
    Config.log_environment_info()
    
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
    
    # Display environment-specific configuration
    training_config = Config.get_training_config_for_environment()
    print(f"- Training config: {training_config}")
    
    if Config.is_kaggle_environment():
        kaggle_config = Config.get_kaggle_config()
        print(f"- Kaggle config: {kaggle_config}")
