from pydantic_settings import BaseSettings
from typing import Dict, List, Optional

import torch

# Класс для хранения переменных среды проекта
# Определены переменные, ожидаемые типы и дефолтные значения
class Settings(BaseSettings):
    DB_NAME: str = 'recsys'
    DB_USER: str = 'postgres'
    DB_PASSWORD: str = 'password'
    DB_HOST: str = 'localhost'
    DB_PORT: int = 5432

    PADDING_IDX: int = 0              # Индекс для паддинга
    RANDOM_SEED: int = 42
    DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS: int = 4

    NUMERICAL_FEATURE_COLUMNS: List[str] = [
        'price', 'session_event_num', 'user_global_event_num',
        'user_views_before', 'user_carts_before', 'user_purchases_before',
        'product_views_before', 'product_purchases_before', 'product_avg_price',
        'category_views_before', 'category_avg_price',
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
        'day_sin', 'day_cos', 'month_sin', 'month_cos', 'year', 'is_weekend',
        'is_holiday', 'days_to_next_holiday', 'days_from_last_holiday'
    ]
    CATEGORICAL_FEATURE_COLUMNS: List[str] = [
        'brand', 'holiday_name'
    ]
    # Параметры Модели и RL
    MAX_HISTORY_LENGTH: int = 20      # Макс. длина истории продуктов для состояния
    MIN_HISTORY_LENGTH: int = 3       # Минимальная длина истории продуктов для перехода
    EMBEDDING_DIM: int = 64           # Размерность эмбеддингов продуктов

    # Награды
    REWARD_MAP: Dict[str, float] = {'view': 0.1, 'cart': 1.0, 'purchase': 5.0}
    DEFAULT_REWARD: float = 0.0
    CATEGORY_MATCH_REWARD: float = 1.0
    # Параметры Обучения
    LEARNING_RATE: float = 1e-6
    GAMMA: float = 0.99
    BATCH_SIZE: int = 128
    EPOCHS: int = 20
    TARGET_UPDATE_FREQ: int = 100
    LOG_FREQ: int = 100

    class Config:
        env_file = ".env"


settings = Settings()

