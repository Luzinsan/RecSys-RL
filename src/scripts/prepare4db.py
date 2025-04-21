import pandas as pd
from pathlib import Path
import holidays
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import save_data, setup_logger


if __name__ == '__main__':
    DATASET_PATH = Path('datasets/ecommerce-events-history-in-electronics-store')
    OUTPUT_PATH = Path('datasets/prepared4db/events_clean.csv')
    HOLIDAYS_PATH = Path('datasets/prepared4db/holidays.csv')
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    os.makedirs(HOLIDAYS_PATH.parent, exist_ok=True)

    logger = setup_logger()

    events = pd.read_csv(
        DATASET_PATH / 'events.csv',
        parse_dates=['event_time'],
        date_format='%Y-%m-%d %H:%M:%S UTC',
        dtype={
            'event_type': 'category',
            'product_id': 'int32',
            'category_id': 'int64',
            'category_code': 'category',
            'brand': 'category',
            'price': 'float32',
            'user_id': 'int64',
            'user_session': 'string'
        }
    )
    logger.info(f'Loaded {len(events)} events')

    logger.info(f'Dropped {events["user_session"].isna().sum()} events with missing user_session')
    events = events.dropna(subset=['user_session'])
    
    logger.info(f'Dropped {events.duplicated(subset=['user_session', 'event_time', 'product_id']).sum()} duplicate events')
    events = events.drop_duplicates(subset=['user_session', 'event_time', 'product_id'])
    
    save_data(events, OUTPUT_PATH, logger=logger)
    logger.info(f'Saved {len(events)} events to {OUTPUT_PATH}')

    holidays_df = pd.DataFrame(holidays.country_holidays(
        'RU', 
        years=list(range(2014, 2026))).items(), 
        columns=['date', 'holiday_name']
    )
    save_data(holidays_df, HOLIDAYS_PATH, logger=logger)
    logger.info(f'Saved {len(holidays_df)} holidays to {HOLIDAYS_PATH}')
