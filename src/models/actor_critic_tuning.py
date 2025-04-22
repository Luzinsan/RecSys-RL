import optuna
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.dataclass import SessionTransitionDataset
from src.models.actor_critic_model import ActorCritic
from src.models.actor_critic_trainer import ActorCriticTrainer
from src.config.configs import settings

def objective(trial: optuna.Trial,
              train_ds: SessionTransitionDataset,
              val_ds:   SessionTransitionDataset,
              num_products, num_brands, num_holidays, num_numerical,
              epochs: int):
    # гиперпараметры
    lr           = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    gamma        = trial.suggest_float("gamma", 0.9, 0.999)
    value_coef   = trial.suggest_float("value_coef", 0.1, 1.0)
    entropy_coef = trial.suggest_float("entropy_coef", 0.0, 0.1)
    batch_size   = trial.suggest_int("batch_size", 32, 256, step=32)
    # модель
    model = ActorCritic(
        num_products, 16, 32, settings.PADDING_IDX,
        num_brands, 8, num_holidays, 4,
        num_numerical, 64
    ).to(settings.DEVICE)
    trainer = ActorCriticTrainer(
        model, optim.Adam, lr,
        gamma, value_coef, entropy_coef, settings.DEVICE
    )
    # даталоадеры
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    # тренировка
    for e in range(epochs):
        trainer.train_epoch(train_loader)
        stats = trainer.evaluate(val_loader)
        trial.report(stats['policy_loss']+stats['value_loss'], e)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return stats['policy_loss']+stats['value_loss']
