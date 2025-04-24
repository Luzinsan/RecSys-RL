import logging
import os
import sys
from pathlib import Path
from typing import Optional
import optuna
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from torch.utils.data import DataLoader
import pandas as pd

from src.models.dataclass import SessionTransitionDataset
from src.models.actor_critic.actor_critic_model import ActorCritic
from src.models.actor_critic.actor_critic_trainer import ActorCriticTrainer
from src.models.evaluate import calculate_all_metrics_and_report
from src.utils.data_utils import setup_logger, setup_seed
from src.utils.pg_connect import PostgresHandler
from src.config.configs import settings



def train_final_model(model_cls, 
                      model_params, 
                      trainer_params, 
                      train_dataloader, 
                      final_epochs, 
                      model_save_path: Optional[str] = None):
    logger.info(f"Initializing model '{model_cls.__name__}'...")
    policy_net = model_cls(**model_params).to(settings.DEVICE)

    trainer = ActorCriticTrainer(
        model=policy_net,
        **trainer_params
    )

    logger.info(f"Starting final training for {final_epochs} epochs...")
    for epoch in range(final_epochs):
        train_stats = trainer.train_epoch(train_dataloader)
        logger.info(f"Final Training Epoch {epoch+1}/{final_epochs} finished. Train Loss: {train_stats['policy_loss']:.4f}, {train_stats['value_loss']:.4f}, {train_stats['entropy']:.4f}")
    logger.info("Final training finished.")

    os.makedirs(Path(model_save_path).parent, exist_ok=True)
    torch.save(policy_net.state_dict(), model_save_path)
    logger.info(f"Model state_dict saved to {model_save_path}")
    
    return policy_net, trainer


class PolicyWrapper(torch.nn.Module):
    def __init__(self, ac_model):
        super().__init__()
        self.ac_model = ac_model
    def forward(self, *args):
        logits, _ = self.ac_model(*args)
        return logits
    
    
class EvalTrainerWrapper:
    def __init__(self, trainer):
        self.trainer = trainer
    def evaluate(self, dataloader):
        stats = self.trainer.evaluate(dataloader)
        
        return stats.get('policy_loss', 0.0) + stats.get('value_loss', 0.0)
    def __getattr__(self, name):
        return getattr(self.trainer, name)
    

class RandomActorCritic(torch.nn.Module):
    def __init__(self, num_products):
        super().__init__()
        self.num_products = num_products
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, state_history, state_length, state_numerical_features, state_brand_idx, state_holiday_idx):
        batch_size = state_history.size(0)
        logits = torch.rand(batch_size, self.num_products, device=settings.DEVICE)
        values = torch.rand(batch_size, device=settings.DEVICE)
        return logits, values
    


if __name__ == '__main__':
    
    MAIN_MODEL_STUDY_NAME = 'RecSys_Actor_critic_val'
    BASELINE_STUDY_NAME = 'RecSys_dqn_recommender_val_with_category_fixed_reward'
    STUDY_DB_PATH = 'sqlite:///optuna_study.db'
    SAVE_DIR = "src/models/actor_critic/trained_models"
    os.makedirs(SAVE_DIR, exist_ok=True)

    REPORT_SAVE_PATH = os.path.join(SAVE_DIR, "evaluation_report.html")
    MAIN_MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "0.pth")
    BASELINE_MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "actor_critic_final.pth")

    FINAL_EPOCHS = 10
    METRICS_K = 10
    DATA_LIMIT = 10000
    TRAIN_SPLIT = 0.85
    
    logger = setup_logger(level=logging.INFO)
    setup_seed(settings.RANDOM_SEED)
    logger.info(f"Running final evaluation script. Device: {settings.DEVICE}")
    
    logger.info(f"Loading study '{MAIN_MODEL_STUDY_NAME}'...")
    study_main = optuna.load_study(study_name=MAIN_MODEL_STUDY_NAME, storage=STUDY_DB_PATH)
    params_main = study_main.best_trial.params
    # params_main['lr'] /= 100
    logger.info(f"Best params for Main Model ({study_main.best_trial.number}): {params_main}")
    
    logger.info(f"Loading study '{BASELINE_STUDY_NAME}'...")
    study_baseline = optuna.load_study(study_name=BASELINE_STUDY_NAME, storage=STUDY_DB_PATH)
    params_baseline = study_baseline.best_trial.params
    params_baseline['lr'] /= 100
    logger.info(f"Best params for Baseline Model ({study_baseline.best_trial.number}): {params_baseline}")
    
    logger.info("Loading data...")
    df = PostgresHandler.send(f"SELECT * FROM e_commerce.events_encoded ORDER BY event_time")
    logger.info(f"Data loaded: {len(df)} rows")
    df['event_time'] = pd.to_datetime(df['event_time'])
    
    n_rows = len(df)
    train_split_idx = int(n_rows * TRAIN_SPLIT)
    df_train = df[:train_split_idx].copy()
    product_to_category_map = (
        df_train[['product_id_idx', 'category_id']]
            .drop_duplicates(subset=['product_id_idx'])
            .set_index('product_id_idx')['category_id']
            .to_dict()
    )
    df_test = df[train_split_idx:].copy()
    logger.info(f"Data split: Train={len(df_train)}, Test={len(df_test)}")
    del df

    
    NUM_PRODUCTS = df_train['product_id_idx'].max() + 1
    NUM_BRANDS = df_train['brand'].max() + 1
    NUM_HOLIDAYS = df_train['holiday_name'].max() + 1
    NUM_NUMERICAL_FEATURES = len(settings.NUMERICAL_FEATURE_COLUMNS)
    logger.info(f"Num products: {NUM_PRODUCTS}, Brands: {NUM_BRANDS}, Holidays: {NUM_HOLIDAYS}")

    
    ds = SessionTransitionDataset(
        df_train,
        numerical_feature_cols=settings.NUMERICAL_FEATURE_COLUMNS,
        categorical_feature_cols=settings.CATEGORICAL_FEATURE_COLUMNS,
        max_history_length=settings.MAX_HISTORY_LENGTH,
        padding_idx=settings.PADDING_IDX,
        reward_map=settings.REWARD_MAP,
        default_reward=settings.DEFAULT_REWARD
    )
    train_loader = DataLoader(ds, batch_size=params_main['batch_size'], shuffle=True, num_workers=8)
    
    ds_test = SessionTransitionDataset(
        df_test,
        numerical_feature_cols=settings.NUMERICAL_FEATURE_COLUMNS,
        categorical_feature_cols=settings.CATEGORICAL_FEATURE_COLUMNS,
        max_history_length=settings.MAX_HISTORY_LENGTH,
        padding_idx=settings.PADDING_IDX,
        reward_map=settings.REWARD_MAP,
        default_reward=settings.DEFAULT_REWARD
    )
    test_loader = DataLoader(ds_test, batch_size=params_main['batch_size'], shuffle=False, num_workers=8)

    
    model_params = {
        'num_products': NUM_PRODUCTS,
        'product_embedding_dim': params_main['product_embedding_dim'],
        'gru_hidden_size': params_main['gru_hidden_size'],
        'padding_idx': settings.PADDING_IDX,
        'num_brands': NUM_BRANDS,
        'brand_embedding_dim': params_main['brand_embedding_dim'],
        'num_holidays': NUM_HOLIDAYS,
        'holiday_embedding_dim': params_main['holiday_embedding_dim'],
        'num_numerical_features': NUM_NUMERICAL_FEATURES,
        'intermediate_layer_size': params_main['intermediate_layer_size']
    }
    trainer_params = {
        'optimizer_cls': torch.optim.AdamW,
        'lr': params_main['lr']/100,
        'gamma': params_main['gamma'],
        'value_coef': params_main['value_coef'],
        'entropy_coef': params_main['entropy_coef'],
        'device': settings.DEVICE
    }

    policy_net, trainer = train_final_model(
        ActorCritic,
        model_params,
        trainer_params,
        train_loader,
        FINAL_EPOCHS,
        MAIN_MODEL_SAVE_PATH
    )

    train_stats = trainer.train_epoch(train_loader)
    print("Train stats:", train_stats)
    eval_stats = trainer.evaluate(train_loader)
    print("Eval stats:", eval_stats)

    policy_model = PolicyWrapper(policy_net)
    eval_trainer = EvalTrainerWrapper(trainer)

    random_ac_model = RandomActorCritic(NUM_PRODUCTS)
    baseline_policy = PolicyWrapper(random_ac_model)
    baseline_ac_trainer_raw = ActorCriticTrainer(
        model=random_ac_model,
        optimizer_cls=torch.optim.AdamW,
        lr=0.0,
        gamma=params_main['gamma'],
        value_coef=params_main['value_coef'],
        entropy_coef=params_main['entropy_coef'],
        device=settings.DEVICE
    )
    baseline_trainer = EvalTrainerWrapper(baseline_ac_trainer_raw)

    metrics_main, metrics_baseline = calculate_all_metrics_and_report(
        policy_net=policy_model,
        trainer=eval_trainer,
        test_dataloader=test_loader,
        test_df=df_test,
        k=METRICS_K,
        settings=settings,
        product_to_category_map=product_to_category_map,
        policy_net_baseline=baseline_policy,
        trainer_baseline=baseline_trainer,
        generate_report=True,
        report_save_path=REPORT_SAVE_PATH
    )
    print("Final Metrics:", metrics_main)
