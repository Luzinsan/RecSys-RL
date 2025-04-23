import optuna
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from src.models.dataclass import SessionTransitionDataset
from src.models.actor_critic_model import ActorCritic
from src.models.actor_critic_trainer import ActorCriticTrainer
from src.config.configs import settings
from src.utils import setup_logger

logger = setup_logger()

def objective(trial: optuna.Trial,
              train_ds: SessionTransitionDataset,
              val_ds: SessionTransitionDataset,
              num_products: int,
              num_brands: int,
              num_holidays: int,
              num_numerical: int,
              epochs: int):
    """
    Enhanced hyperparameter optimization objective with:
    - More comprehensive parameter ranges
    - Early stopping
    - Better logging
    - Memory management
    """
    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    value_coef = trial.suggest_float("value_coef", 0.1, 1.0)
    entropy_coef = trial.suggest_float("entropy_coef", 0.01, 0.1)
    batch_size = trial.suggest_int("batch_size", 32, 512, step=32)
    
    # Model architecture parameters
    product_embedding_dim = trial.suggest_int("product_embedding_dim", 16, 64, step=8)
    gru_hidden_size = trial.suggest_int("gru_hidden_size", 32, 128, step=32)
    brand_embedding_dim = trial.suggest_int("brand_embedding_dim", 8, 32, step=8)
    holiday_embedding_dim = trial.suggest_int("holiday_embedding_dim", 4, 16, step=4)
    intermediate_layer_size = trial.suggest_int("intermediate_layer_size", 64, 256, step=64)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    
    logger.info(f"[Trial {trial.number}] Starting with params:")
    logger.info(f"  Learning rate: {lr:.2e}")
    logger.info(f"  Gamma: {gamma:.3f}")
    logger.info(f"  GAE Lambda: {gae_lambda:.3f}")
    logger.info(f"  Value coefficient: {value_coef:.3f}")
    logger.info(f"  Entropy coefficient: {entropy_coef:.3f}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Model architecture:")
    logger.info(f"    Product embedding dim: {product_embedding_dim}")
    logger.info(f"    GRU hidden size: {gru_hidden_size}")
    logger.info(f"    Brand embedding dim: {brand_embedding_dim}")
    logger.info(f"    Holiday embedding dim: {holiday_embedding_dim}")
    logger.info(f"    Intermediate layer size: {intermediate_layer_size}")
    logger.info(f"    Dropout rate: {dropout_rate}")
    
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
        num_numerical_features=num_numerical,
        intermediate_layer_size=intermediate_layer_size,
        dropout_rate=dropout_rate
    ).to(settings.DEVICE)
    
    # Initialize trainer
    trainer = ActorCriticTrainer(
        model=model,
        optimizer_cls=optim.AdamW,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=0.5,
        device=settings.DEVICE
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if settings.DEVICE == 'cuda' else False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True if settings.DEVICE == 'cuda' else False
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    trial_start_time = time.time()
    
    for epoch in range(epochs):
        # Train
        train_stats = trainer.train_epoch(train_loader)
        
        # Evaluate
        val_stats = trainer.evaluate(val_loader)
        val_loss = val_stats['total_loss']
        
        # Log progress
        logger.info(f"[Trial {trial.number}] Epoch {epoch+1}/{epochs}")
        logger.info(f"  Train - Policy Loss: {train_stats['policy_loss']:.4f}, "
                   f"Value Loss: {train_stats['value_loss']:.4f}, "
                   f"Entropy: {train_stats['entropy']:.4f}")
        logger.info(f"  Val   - Policy Loss: {val_stats['policy_loss']:.4f}, "
                   f"Value Loss: {val_stats['value_loss']:.4f}, "
                   f"Entropy: {val_stats['entropy']:.4f}")
        
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
