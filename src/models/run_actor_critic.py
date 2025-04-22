import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.models.dataclass import SessionTransitionDataset
from src.models.actor_critic_model import ActorCritic
from src.models.actor_critic_trainer import ActorCriticTrainer
from src.models.evaluate import calculate_all_metrics_and_report
from src.config.configs import settings


def generate_synthetic_events(
    num_users: int = 200,
    seq_len: int = 15,
    num_products: int = 50,
    num_brands: int = 5,
    num_holidays: int = 4,
    num_numerical: int = 3
) -> pd.DataFrame:
    """
    Генерирует DataFrame синтетических событий для пользователей.
    """
    records = []
    now = datetime.utcnow()
    event_types = ['view', 'click', 'purchase']

    for u in range(num_users):
        user_id = f"user_{u}"
        session_id = f"sess_{u}_0"
        t = now
        for i in range(seq_len):
            prod = np.random.randint(0, num_products)
            ev = np.random.choice(event_types, p=[0.7, 0.25, 0.05])
            price = float(np.random.randint(10, 200)) if ev == 'purchase' else 0.0
            brand = np.random.randint(0, num_brands)
            hol = np.random.randint(0, num_holidays)
            nums = np.random.rand(num_numerical)
            row = {
                'user_id': user_id,
                'user_session': session_id,
                'event_time': t,
                'product_id_idx': prod,
                'event_type': ev,
                'price': price,
                'brand': brand,
                'holiday_name': hol
            }
            # числовые фичи
            for j in range(num_numerical):
                row[f'num_feat_{j}'] = float(nums[j])
            records.append(row)
            t += timedelta(seconds=1)
    return pd.DataFrame(records)


def main():
    # --- 1. Параметры ---
    NUM_PRODUCTS = 50
    BRAND_DIM    = 5
    HOLIDAY_DIM  = 4
    NUM_NUMERIC  = 3
    MAX_HISTORY  = 5
    PADDING_IDX  = 0
    REWARD_MAP   = {'view': 0.0, 'click': 1.0, 'purchase': 5.0}
    DEFAULT_R    = 0.0
    TOP_K        = 5

    # --- 2. Синтетические данные ---
    df = generate_synthetic_events(
        num_users=200,
        seq_len=15,
        num_products=NUM_PRODUCTS,
        num_brands=BRAND_DIM,
        num_holidays=HOLIDAY_DIM,
        num_numerical=NUM_NUMERIC
    )

    # --- 3. Dataset и DataLoader ---
    num_cols = [f'num_feat_{j}' for j in range(NUM_NUMERIC)]
    ds = SessionTransitionDataset(
        df,
        numerical_feature_cols=num_cols,
        categorical_feature_cols=[],
        max_history_length=MAX_HISTORY,
        padding_idx=PADDING_IDX,
        reward_map=REWARD_MAP,
        default_reward=DEFAULT_R
    )
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)

    # --- 4. Модель и тренер ---
    model = ActorCritic(
        num_products=NUM_PRODUCTS,
        product_embedding_dim=16,
        gru_hidden_size=32,
        padding_idx=PADDING_IDX,
        num_brands=BRAND_DIM,
        brand_embedding_dim=8,
        num_holidays=HOLIDAY_DIM,
        holiday_embedding_dim=8,
        num_numerical_features=NUM_NUMERIC,
        intermediate_layer_size=64
    )
    trainer = ActorCriticTrainer(
        model=model,
        optimizer_cls=torch.optim.Adam,
        lr=1e-3,
        gamma=0.99,
        value_coef=0.5,
        entropy_coef=0.01,
        device=settings.DEVICE
    )

    # --- 5. Обучение и первичная оценка ---
    train_stats = trainer.train_epoch(loader)
    print("Train stats:", train_stats)
    eval_stats  = trainer.evaluate(loader)
    print("Eval stats:", eval_stats)

    # --- 6. Обёртки для evaluate.py ---
    # policy_net должен возвращать только logits
    class PolicyWrapper(torch.nn.Module):
        def __init__(self, ac_model):
            super().__init__()
            self.ac_model = ac_model
        def forward(self, *args):
            logits, _ = self.ac_model(*args)
            return logits

    policy_model = PolicyWrapper(model)

    # trainer.evaluate должен возвращать float loss
    class EvalTrainerWrapper:
        def __init__(self, trainer):
            self.trainer = trainer
        def evaluate(self, dataloader):
            stats = self.trainer.evaluate(dataloader)
            # объединяем policy+value loss
            return stats.get('policy_loss', 0.0) + stats.get('value_loss', 0.0)
        def __getattr__(self, name):
            return getattr(self.trainer, name)

    eval_trainer = EvalTrainerWrapper(trainer)

    # --- 7. Полная оценка и отчёт ---
    metrics_main, metrics_baseline = calculate_all_metrics_and_report(
        policy_net=policy_model,
        trainer=eval_trainer,
        test_dataloader=loader,
        test_df=df,
        k=TOP_K,
        settings=settings,
        policy_net_baseline=policy_model,
        trainer_baseline=eval_trainer,
        generate_report=True,
        report_save_path="ac_evaluation_report.html"
    )
    print("Final Metrics:", metrics_main)


if __name__ == "__main__":
    main()
