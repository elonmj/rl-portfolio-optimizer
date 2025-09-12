from train import PortfolioTrainer
import warnings
warnings.filterwarnings('ignore')

# Test avec seulement 5 épisodes
config_overrides = {
    'MAX_EPISODES': 5,
    'EVAL_FREQUENCY': 2,
    'SAVE_FREQUENCY': 10,
    'BATCH_SIZE': 32,
}

trainer = PortfolioTrainer(config_overrides)
print('🧪 Test d\'entraînement avec 5 épisodes...')

try:
    metrics = trainer.train(num_episodes=5)
    print('✅ Test d\'entraînement réussi!')
    if metrics['total_return']:
        final_return = metrics['total_return'][-1]
        print(f'Retour final: {final_return:.2%}')
except Exception as e:
    print(f'❌ Erreur: {e}')
    import traceback
    traceback.print_exc()
