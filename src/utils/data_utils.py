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
    Setup logger
    
    Args:
        name: Logger name
        level: Logging level
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
    Load data from file in csv, parquet, pickle, excel format.
    
    Args:
        input_path: Path to data file
        logger: Logger object (optional)
    """
    if logger is None:
        logger = setup_logger()
    
    logger.info(f"Loading data from {input_path}")
    
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
    Save DataFrame to file.
    
    Args:
        df: pandas.DataFrame to save
        output_path: Path to save file
        index: Include index when saving
        logger: Logger object (optional)
    """
    if logger is None:
        logger = setup_logger()
    logger.info(f"Saving data to {output_path}")
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
            raise ValueError(f"Unsupported file format: {file_ext}")
        logger.info(f"Data saved successfully. Size: {df.shape}")
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise 
