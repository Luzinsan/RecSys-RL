import pandas as pd
import logging
import os
import traceback
import torch
import random
import numpy as np

def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def pad_collate_fn(batch):
    keys = batch[0].keys()
    collated_batch = {}
    for key in keys:
        collated_batch[key] = torch.stack([item[key] for item in batch])
    return collated_batch


def setup_logger(name=__name__, 
                 level=logging.INFO):
    """
    Настройка логгера
    
    Args:
        name: Имя логгера
        level: Уровень логирования
    """
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    
    return logger

def load_data(input_path, 
              logger=None):
    """
    Загрузка данных из файла в формате csv, parquet, pickle, excel.
    
    Args:
        input_path: Путь к файлу данных
        logger: Объект логгера (опционально)
    """
    if logger is None:
        logger = setup_logger()
    
    logger.info(f"Загрузка данных из {input_path}")
    
    try:
        file_ext = os.path.splitext(input_path)[-1].lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(input_path)
        elif file_ext == '.parquet':
            df = pd.read_parquet(input_path)
        elif file_ext == '.pickle' or file_ext == '.pkl':
            df = pd.read_pickle(input_path)
        elif file_ext == '.xlsx' or file_ext == '.xls':
            df = pd.read_excel(input_path)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {file_ext}")
        
        logger.info(f"Данные успешно загружены. Форма данных: {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def save_data(df: pd.DataFrame, 
              output_path: str, 
              index: bool = False, 
              logger: logging.Logger = None):
    """
    Сохранение DataFrame в файл.
    
    Args:
        df: pandas.DataFrame для сохранения
        output_path: Путь для сохранения файла
        index: Включать ли индекс при сохранении
        logger: Объект логгера (опционально)
    """
    if logger is None:
        logger = setup_logger()
    logger.info(f"Сохранение данных в {output_path}")
    try:
        file_ext = os.path.splitext(output_path)[1].lower()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if file_ext == '.csv':
            df.to_csv(output_path, index=index)
        elif file_ext == '.parquet':
            df.to_parquet(output_path, index=index)
        elif file_ext == '.pickle' or file_ext == '.pkl':
            df.to_pickle(output_path)
        elif file_ext == '.xlsx' or file_ext == '.xls':
            df.to_excel(output_path, index=index)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {file_ext}")
        logger.info(f"Данные успешно сохранены. Размер: {df.shape}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении данных: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise 
