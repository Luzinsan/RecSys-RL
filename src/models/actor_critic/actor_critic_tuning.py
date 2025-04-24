import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import optuna
from src.models.dataclass import SessionTransitionDataset
from src.models.actor_critic.actor_critic_model import ActorCritic
from src.models.actor_critic.actor_critic_trainer import ActorCriticTrainer
from src.config.configs import settings
from src.utils import PostgresHandler, setup_logger, pad_collate_fn, setup_seed

logger = setup_logger()

def objective(trial: optuna.Trial, 
              train_dataset: SessionTransitionDataset,
              val_dataset: SessionTransitionDataset,
              num_products, 
              num_brands, 
              num_holidays, 
              num_numerical_features, 
              epochs_per_trial):
    """
    Objective function for Optuna hyperparameter optimization.
    Trains Actor-Critic model with given hyperparameters and returns validation loss.
    """
    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 64, 512, step=64)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    value_coef = trial.suggest_float("value_coef", 0.3, 0.7)
    entropy_coef = trial.suggest_float("entropy_coef", 0.005, 0.05)
    
    # Model architecture parameters
    product_embedding_dim = trial.suggest_int("product_embedding_dim", 32, 128, step=32)
    gru_hidden_size = trial.suggest_int("gru_hidden_size", 64, 256, step=64)
    brand_embedding_dim = trial.suggest_int("brand_embedding_dim", 16, 64, step=16)
    holiday_embedding_dim = trial.suggest_int("holiday_embedding_dim", 8, 32, step=8)
    intermediate_layer_size = trial.suggest_int("intermediate_layer_size", 128, 512, step=128)

    logger.info(f"[Trial {trial.number}] Starting with params:")
    logger.info(f"  Learning rate: {lr:.2e}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Gamma: {gamma:.3f}")
    logger.info(f"  Value coefficient: {value_coef:.3f}")
    logger.info(f"  Entropy coefficient: {entropy_coef:.3f}")
    logger.info(f"  Model architecture:")
    logger.info(f"    Product embedding dim: {product_embedding_dim}")
    logger.info(f"    GRU hidden size: {gru_hidden_size}")
    logger.info(f"    Brand embedding dim: {brand_embedding_dim}")
    logger.info(f"    Holiday embedding dim: {holiday_embedding_dim}")
    logger.info(f"    Intermediate layer size: {intermediate_layer_size}")
    
    trial_start_time = time.time()
    setup_seed(settings.RANDOM_SEED + trial.number)
   
    # Initialize model
    model = ActorCritic(
        num_products=num_products,
        product_embedding_dim=product_embedding_dim,
        gru_hidden_size=gru_hidden_size,
        padding_idx=settings.PADDING_IDX,
        num_brands=num_brands,
        brand_embedding_dim=brand_embedding_dim,
        num_holidays=num_holidays,
        holiday_embedding_dim=holiday_embedding_dim,
        num_numerical_features=num_numerical_features,
        intermediate_layer_size=intermediate_layer_size
    ).to(settings.DEVICE)
    
    # Initialize trainer
    trainer = ActorCriticTrainer(
        model=model,
        optimizer_cls=optim.AdamW,
        lr=lr,
        gamma=gamma,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        device=settings.DEVICE
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
        num_workers=2,
        pin_memory=True if settings.DEVICE == 'cuda' else False
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=pad_collate_fn,
        num_workers=2,
        pin_memory=True if settings.DEVICE == 'cuda' else False
    )

    # Training loop
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs_per_trial):
        # Train
        train_stats = trainer.train_epoch(train_dataloader)
        
        # Evaluate
        val_stats = trainer.evaluate(val_dataloader)
        val_loss = val_stats['policy_loss'] + val_stats['value_loss']
        
        # Log progress
        logger.info(f"[Trial {trial.number}] Epoch {epoch+1}/{epochs_per_trial}")
        logger.info(f"  Train - Policy Loss: {train_stats['policy_loss']:.4f}, "
                   f"Value Loss: {train_stats['value_loss']:.4f}, "
                   f"Entropy: {train_stats['entropy']:.4f}")
        logger.info(f"  Val   - Policy Loss: {val_stats['policy_loss']:.4f}, "
                   f"Value Loss: {val_stats['value_loss']:.4f}")
        
        # Report to Optuna
        trial.report(val_loss, epoch)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"[Trial {trial.number}] Early stopping triggered at epoch {epoch+1}")
                break
        
        # Pruning
        if trial.should_prune():
            trial_duration = time.time() - trial_start_time
            logger.info(f"[Trial {trial.number}] Pruned after epoch {epoch+1}. "
                       f"Duration: {trial_duration:.2f}s")
            raise optuna.TrialPruned()
    
    trial_duration = time.time() - trial_start_time
    logger.info(f"[Trial {trial.number}] Completed in {trial_duration:.2f}s. "
               f"Best validation loss: {best_val_loss:.4f}")
    
    return best_val_loss


if __name__ == '__main__':
    logger.info(f"Using device: {settings.DEVICE}")

    logger.info("Loading data...")
    df = pd.read_csv('datasets/events_encoded.csv')
    logger.info(f"Data loaded: {len(df)} rows")

    df['event_time'] = pd.to_datetime(df['event_time'])
    n_rows = len(df)
    train_split_idx = int(n_rows * 0.7)
    val_split_idx = int(n_rows * 0.85)  # 70% + 15%

    df_train = df[:train_split_idx].copy()
    df_val = df[train_split_idx:val_split_idx].copy()
    df_test = df[val_split_idx:].copy()

    logger.info(f"Data split: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    logger.info(f"Time range Train: {df_train['event_time'].min()} - {df_train['event_time'].max()}")
    logger.info(f"Time range Val:   {df_val['event_time'].min()} - {df_val['event_time'].max()}")
    logger.info(f"Time range Test:  {df_test['event_time'].min()} - {df_test['event_time'].max()}")

    NUM_PRODUCTS = df_train['product_id_idx'].max() + 1
    logger.info(f"Num products (for Embedding): {NUM_PRODUCTS}")

    NUM_BRANDS = df_train['brand'].max() + 1
    logger.info(f"Num brands (for Embedding): {NUM_BRANDS}")

    NUM_HOLIDAYS = df_train['holiday_name'].max() + 1
    logger.info(f"Num holidays (for Embedding): {NUM_HOLIDAYS}")

    # --- Create Datasets ---
    logger.info("Preparing train dataset...")
    train_dataset = SessionTransitionDataset(
        df_train, settings.NUMERICAL_FEATURE_COLUMNS, settings.CATEGORICAL_FEATURE_COLUMNS,
        settings.MAX_HISTORY_LENGTH, settings.PADDING_IDX, settings.REWARD_MAP,
        settings.DEFAULT_REWARD, settings.MIN_HISTORY_LENGTH
    )
    logger.info(f"Train dataset prepared with {len(train_dataset)} transitions.")
    del df_train

    logger.info("Preparing validation dataset...")
    val_dataset = SessionTransitionDataset(
        df_val, settings.NUMERICAL_FEATURE_COLUMNS, settings.CATEGORICAL_FEATURE_COLUMNS,
        settings.MAX_HISTORY_LENGTH, settings.PADDING_IDX, settings.REWARD_MAP,
        settings.DEFAULT_REWARD, settings.MIN_HISTORY_LENGTH
    )
    logger.info(f"Validation dataset prepared with {len(val_dataset)} transitions.")
    del df_val, df_test

    N_TRIALS = None  # Set to None for unlimited trials or a specific number
    EPOCHS_PER_TRIAL = 10

    logger.info(f"Starting Optuna study with {N_TRIALS} trials, {EPOCHS_PER_TRIAL} epochs per trial.")

    study = optuna.create_study(
        storage='sqlite:///optuna_study.db', 
        load_if_exists=True, 
        study_name='RecSys_AC_val', 
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner()
    )
    try:
        study.optimize(lambda trial: objective(
            trial, 
            train_dataset,
            val_dataset, 
            NUM_PRODUCTS, 
            NUM_BRANDS, 
            NUM_HOLIDAYS, 
            len(settings.NUMERICAL_FEATURE_COLUMNS), 
            EPOCHS_PER_TRIAL
        ), n_trials=N_TRIALS)
    except KeyboardInterrupt:
        logger.warning("Optuna optimization stopped manually.")

    logger.info("Optuna study finished.")
    if study.best_trial:
        logger.info(f"Best trial number: {study.best_trial.number}")
        logger.info(f"\tBest value (min val loss): {study.best_trial.value:.6f}")
        logger.info("\tBest parameters:")
        for key, value in study.best_trial.params.items():
            logger.info(f"\t\t{key}: {value}")
    else:
        logger.info("No successful trials completed.")
