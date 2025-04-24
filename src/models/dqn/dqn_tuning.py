import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils import PostgresHandler, setup_logger, pad_collate_fn, setup_seed
import optuna
import time
from src.config.configs import settings
from src.models.dataclass import SessionTransitionDataset
from src.models.dqn.dqn_model import DQNRecommender
from src.models.dqn.dqn_trainer import DQLTrainer
from src.models.dqn.baseline import DQNBaseline
logger = setup_logger()



def objective(trial: optuna.Trial, 
              train_dataset: SessionTransitionDataset,
              val_dataset: SessionTransitionDataset,
              product_to_category_map: dict,
              num_products, 
              num_brands, 
              num_holidays, 
              num_numerical_features, 
              epochs_per_trial,
              model):
    
    lr = trial.suggest_float("lr", 1e-7, 1e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 64, 1024, step=64)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    target_update_freq = trial.suggest_int("target_update_freq", 50, 500, step=50)
    gru_hidden_size = trial.suggest_int("gru_hidden_size", 64, 256, step=32)
    intermediate_layer_size = trial.suggest_int("intermediate_layer_size", 64, 256, step=32)
    brand_embedding_dim = trial.suggest_int("brand_embedding_dim", 8, 200, step=8)
    holiday_embedding_dim = trial.suggest_int("holiday_embedding_dim", 2, 5, step=1)

    logger.info(f"[Trial {trial.number}] Starting with params: lr={lr:.2e}, batch_size={batch_size}, gamma={gamma:.3f}, "
                f"target_update={target_update_freq}, gru_hidden={gru_hidden_size}, intermediate={intermediate_layer_size}, "
                f"brand_emb={brand_embedding_dim}, holiday_emb={holiday_embedding_dim}")
    
    trial_start_time = time.time()
    setup_seed(settings.RANDOM_SEED + trial.number)
   
    
    if model=='baseline':
        trainer = DQLTrainer(
            policy_net=DQNBaseline(
                num_products=num_products, product_embedding_dim=gru_hidden_size, gru_hidden_size=gru_hidden_size,
                padding_idx=settings.PADDING_IDX, num_brands=num_brands, brand_embedding_dim=brand_embedding_dim,
                num_holidays=num_holidays, holiday_embedding_dim=holiday_embedding_dim,
                num_numerical_features=num_numerical_features, intermediate_layer_size=intermediate_layer_size
            ).to(settings.DEVICE), 
            target_net=DQNBaseline(
                num_products=num_products, product_embedding_dim=gru_hidden_size, gru_hidden_size=gru_hidden_size,
                padding_idx=settings.PADDING_IDX, num_brands=num_brands, brand_embedding_dim=brand_embedding_dim,
                num_holidays=num_holidays, holiday_embedding_dim=holiday_embedding_dim,
                num_numerical_features=num_numerical_features, intermediate_layer_size=intermediate_layer_size
            ).to(settings.DEVICE), 
            criterion=nn.SmoothL1Loss(), 
            optimizer=optim.AdamW, 
            lr=lr,
            gamma=gamma, 
            target_update_freq=target_update_freq,
            product_to_category_map=product_to_category_map,
            device=settings.DEVICE
        )
    elif model=='dqn_recommender':
        trainer = DQLTrainer(
            policy_net=DQNRecommender(
                num_products=num_products, product_embedding_dim=gru_hidden_size, gru_hidden_size=gru_hidden_size,
                padding_idx=settings.PADDING_IDX, num_brands=num_brands, brand_embedding_dim=brand_embedding_dim,
                num_holidays=num_holidays, holiday_embedding_dim=holiday_embedding_dim,
                num_numerical_features=num_numerical_features, intermediate_layer_size=intermediate_layer_size
            ).to(settings.DEVICE), 
            target_net=DQNRecommender(
                num_products=num_products, product_embedding_dim=gru_hidden_size, gru_hidden_size=gru_hidden_size,
                padding_idx=settings.PADDING_IDX, num_brands=num_brands, brand_embedding_dim=brand_embedding_dim,
                num_holidays=num_holidays, holiday_embedding_dim=holiday_embedding_dim,
                num_numerical_features=num_numerical_features, intermediate_layer_size=intermediate_layer_size
            ).to(settings.DEVICE), 
            criterion=nn.SmoothL1Loss(), 
            optimizer=optim.AdamW, 
            lr=lr,
            gamma=gamma, 
            target_update_freq=target_update_freq,
            product_to_category_map=product_to_category_map,
            device=settings.DEVICE
        )



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

    for epoch in range(epochs_per_trial):
        train_loss = trainer.train_epoch(train_dataloader)
        final_val_loss = trainer.evaluate(val_dataloader)
        logger.info(f"[Trial {trial.number}] Epoch {epoch+1}/{epochs_per_trial} finished. Train Loss: {train_loss:.4f}, Val Loss: {final_val_loss:.4f}")
        trial.report(final_val_loss, epoch)
        if trial.should_prune():
            trial_duration = time.time() - trial_start_time
            logger.info(f"[Trial {trial.number}] Pruned after epoch {epoch+1}. Duration: {trial_duration:.2f}s")
            raise optuna.TrialPruned()

    return final_val_loss



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='baseline')
    args = parser.parse_args()

    logger.info(f"Using device: {settings.DEVICE}")

    logger.info("Loading data...")
    df = PostgresHandler.send(f"SELECT * FROM e_commerce.events_encoded ORDER BY user_id, event_time LIMIT 100000")
    logger.info(f"Data loaded: {len(df)} rows")

    df['event_time'] = pd.to_datetime(df['event_time'])
    n_rows = len(df)
    train_split_idx = int(n_rows * 0.7)
    val_split_idx = int(n_rows * 0.85) # 70% + 15%

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
    print(df_train.columns)
    product_to_category_map = df_train.drop_duplicates(
        subset=['product_id_idx'])\
            [['product_id_idx', 'category_id']]\
                .set_index('product_id_idx')\
                .to_dict()['category_id']
    
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

    N_TRIALS = 200
    EPOCHS_PER_TRIAL = 10

    logger.info(f"Starting Optuna study with {N_TRIALS} trials, {EPOCHS_PER_TRIAL} epochs per trial.")

    study = optuna.create_study(
        storage='sqlite:///optuna_study.db', 
        load_if_exists=True, 
        study_name=f'RecSys_{args.model}_val_with_category_fixed_reward', 
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )
    try:
        study.optimize(lambda trial: objective(
            trial, 
            train_dataset,
            val_dataset, 
            product_to_category_map,
            NUM_PRODUCTS, 
            NUM_BRANDS, 
            NUM_HOLIDAYS, 
            len(settings.NUMERICAL_FEATURE_COLUMNS), 
            EPOCHS_PER_TRIAL,
            args.model
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
