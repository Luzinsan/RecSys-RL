import torch
import sys
from pathlib import Path
import pandas as pd
import optuna
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import logging
import os
from typing import Optional
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import PostgresHandler, setup_logger, setup_seed, pad_collate_fn
from src.config.configs import settings
from src.models.dataclass import SessionTransitionDataset
from src.models.dqn_trainer import DQLTrainer
from src.models.dqn_model import DQNRecommender
from src.models.baseline import DQNBaseline
from src.models.evaluate import calculate_all_metrics_and_report


def train_final_model(model_cls, 
                      model_params, 
                      trainer_params, 
                      train_dataloader, 
                      final_epochs, 
                      model_save_path: Optional[str] = None):
    logger.info(f"Initializing model '{model_cls.__name__}'...")
    policy_net = model_cls(**model_params).to(settings.DEVICE)
    target_net = model_cls(**model_params).to(settings.DEVICE)

    trainer = DQLTrainer(
        policy_net=policy_net,
        target_net=target_net,
        **trainer_params
    )

    logger.info(f"Starting final training for {final_epochs} epochs...")
    for epoch in range(final_epochs):
        epoch_train_loss = trainer.train_epoch(train_dataloader)
        logger.info(f"Final Training Epoch {epoch+1}/{final_epochs} finished. Train Loss: {epoch_train_loss:.4f}")
    logger.info("Final training finished.")

    os.makedirs(Path(model_save_path).parent, exist_ok=True)
    torch.save(policy_net.state_dict(), model_save_path)
    logger.info(f"Model state_dict saved to {model_save_path}")
    
    return policy_net, trainer



if __name__ == '__main__':
    # --- Настройки
    MAIN_MODEL_STUDY_NAME = 'RecSys_dqn_recommender_val'
    BASELINE_STUDY_NAME = 'RecSys_baseline_val'
    STUDY_DB_PATH = 'sqlite:///optuna_study.db'
    SAVE_DIR = "src/models/trained_models"
    os.makedirs(SAVE_DIR, exist_ok=True)

    REPORT_SAVE_PATH = os.path.join(SAVE_DIR, "evaluation_report.html")
    MAIN_MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "dqn_recommender_final.pth")
    BASELINE_MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "dqn_baseline_final.pth")


    FINAL_EPOCHS = 10
    METRICS_K = 10
    DATA_LIMIT = 10000
    TRAIN_SPLIT = 0.85
    
    
    logger = setup_logger(level=logging.INFO)
    setup_seed(settings.RANDOM_SEED)
    logger.info(f"Running final evaluation script. Device: {settings.DEVICE}")

    # --- 1. Загрузка лучших параметров ---
    logger.info(f"Loading study '{MAIN_MODEL_STUDY_NAME}'...")
    study_main = optuna.load_study(study_name=MAIN_MODEL_STUDY_NAME, storage=STUDY_DB_PATH)
    params_main = study_main.best_trial.params
    params_main['lr'] /= 5
    logger.info(f"Best params for Main Model ({study_main.best_trial.number}): {params_main}")
    
    logger.info(f"Loading study '{BASELINE_STUDY_NAME}'...")
    study_baseline = optuna.load_study(study_name=BASELINE_STUDY_NAME, storage=STUDY_DB_PATH)
    params_baseline = study_baseline.best_trial.params
    params_baseline['lr'] /= 10
    logger.info(f"Best params for Baseline Model ({study_baseline.best_trial.number}): {params_baseline}")
    
    
    # --- 2. Загрузка и подготовка данных ---
    logger.info("Loading data...")
    df = PostgresHandler.send(f"SELECT * FROM e_commerce.events_encoded ORDER BY event_time")
    logger.info(f"Data loaded: {len(df)} rows")
    df['event_time'] = pd.to_datetime(df['event_time'])
    # --- 3. Разделение на Train/Test ---
    n_rows = len(df)
    train_split_idx = int(n_rows * TRAIN_SPLIT)
    df_train = df[:train_split_idx].copy()
    df_test = df[train_split_idx:].copy()
    logger.info(f"Data split: Train={len(df_train)}, Test={len(df_test)}")
    del df

    # --- 4. Определение размеров словарей (по TRAIN) ---
    NUM_PRODUCTS = df_train['product_id_idx'].max() + 1
    NUM_BRANDS = df_train['brand'].max() + 1
    NUM_HOLIDAYS = df_train['holiday_name'].max() + 1
    NUM_NUMERICAL_FEATURES = len(settings.NUMERICAL_FEATURE_COLUMNS)
    logger.info(f"Num products: {NUM_PRODUCTS}, Brands: {NUM_BRANDS}, Holidays: {NUM_HOLIDAYS}")

    # --- 5. Создание Datasets  ---
    dataset_creation_params = {
        'numerical_feature_cols': settings.NUMERICAL_FEATURE_COLUMNS,
        'categorical_feature_cols': settings.CATEGORICAL_FEATURE_COLUMNS,
        'max_history_length': settings.MAX_HISTORY_LENGTH,
        'padding_idx': settings.PADDING_IDX,
        'reward_map': settings.REWARD_MAP,
        'default_reward': settings.DEFAULT_REWARD,
        'min_history_length': settings.MIN_HISTORY_LENGTH
    }
    logger.info("Preparing train dataset...")
    train_dataset = SessionTransitionDataset(df_train, **dataset_creation_params)
    logger.info(f"Train dataset prepared with {len(train_dataset)} transitions.")
    del df_train

    logger.info("Preparing test dataset...")
    test_dataset = SessionTransitionDataset(df_test, **dataset_creation_params)
    logger.info(f"Test dataset prepared with {len(test_dataset)} transitions.")

    # --- 6. Подготовка параметров для моделей и тренера ---
    common_model_params = {
        'num_products': NUM_PRODUCTS,
        'padding_idx': settings.PADDING_IDX,
        'num_brands': NUM_BRANDS,
        'num_holidays': NUM_HOLIDAYS,
        'num_numerical_features': NUM_NUMERICAL_FEATURES,
    }
    common_trainer_params = {
        'criterion': nn.SmoothL1Loss(),
        'optimizer': optim.AdamW,
    }
    # Baseline
    model_params_baseline = {
        **common_model_params,
        'product_embedding_dim': params_baseline['gru_hidden_size'],
        'gru_hidden_size': params_baseline['gru_hidden_size'],
        'brand_embedding_dim': params_baseline['brand_embedding_dim'],
        'holiday_embedding_dim': params_baseline['holiday_embedding_dim'],
        'intermediate_layer_size': params_baseline['intermediate_layer_size']
    }
    trainer_params_baseline = {
        **common_trainer_params,
        'lr': params_baseline['lr'],
        'gamma': params_baseline['gamma'],
        'target_update_freq': params_baseline['target_update_freq']
    }
    # Основная модель
    model_params_main = {
        **common_model_params,
        'product_embedding_dim': params_main['gru_hidden_size'],
        'gru_hidden_size': params_main['gru_hidden_size'],
        'brand_embedding_dim': params_main['brand_embedding_dim'],
        'holiday_embedding_dim': params_main['holiday_embedding_dim'],
        'intermediate_layer_size': params_main['intermediate_layer_size']
    }
    trainer_params_main = {
        **common_trainer_params,
        'lr': params_main['lr'],
        'gamma': params_main['gamma'],
        'target_update_freq': params_main['target_update_freq']
    }
    
    # --- 7. Создание DataLoaders  ---
    dataloader_creation_params = {
        'collate_fn': pad_collate_fn,
        'num_workers': settings.NUM_WORKERS,
        'pin_memory': settings.DEVICE.type == 'cuda'
    }
    train_batch_size = params_main.get('batch_size', 128)
    test_batch_size = train_batch_size * 2
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, **dataloader_creation_params)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **dataloader_creation_params)

    # --- 8. Обучение Baseline Модели  ---
    logger.info("="*20 + " Training Baseline Model " + "="*20)
    policy_net_baseline, trainer_baseline = train_final_model(
        DQNBaseline, model_params_baseline, trainer_params_baseline, train_dataloader, FINAL_EPOCHS,
        model_save_path=BASELINE_MODEL_SAVE_PATH
    )

    # --- 9. Обучение Основной Модели  ---
    logger.info("="*20 + " Training Main Model " + "="*20)
    policy_net_main, trainer_main = train_final_model(
        DQNRecommender, model_params_main, trainer_params_main, train_dataloader, FINAL_EPOCHS,
        model_save_path=MAIN_MODEL_SAVE_PATH
    )


    # --- 10. Оценка и Генерация Отчета ---
    logger.info("="*20 + " Evaluating Models and Generating Report " + "="*20)
    metrics_main_final, metrics_baseline_final = calculate_all_metrics_and_report(
        policy_net=policy_net_main,
        trainer=trainer_main,
        test_dataloader=test_dataloader,
        test_df=df_test,
        k=METRICS_K,
        settings=settings,
        policy_net_baseline=policy_net_baseline,
        trainer_baseline=trainer_baseline,
        generate_report=True,
        report_save_path=REPORT_SAVE_PATH
    )

    # --- 11. Вывод сравнения в лог---
    logger.info("="*20 + " Final Comparison (Log) " + "="*20)
    if metrics_baseline_final:
        all_metric_keys = sorted(list(set(metrics_main_final.keys()) | set(metrics_baseline_final.keys())))
        comparison_data = {
            'Main': [metrics_main_final.get(key, float('nan')) for key in all_metric_keys],
            'Baseline': [metrics_baseline_final.get(key, float('nan')) for key in all_metric_keys]
        }
        comparison_df = pd.DataFrame(comparison_data, index=all_metric_keys)
        comparison_df = comparison_df.fillna('N/A')
        logger.info("\n" + comparison_df.round(6).to_string())
    else:
         logger.info("Baseline metrics not available for comparison in log.")

    logger.info("Evaluation script finished.")
