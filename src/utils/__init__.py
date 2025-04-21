import sys
from pathlib import Path

def find_project_root():
    path = Path(__file__).resolve()
    while not (path / 'pyproject.toml').exists():
        path = path.parent
        if path == path.parent:
            break
    return path


sys.path.insert(0, str(find_project_root()))

from src.utils.data_utils import (
    setup_logger, 
    load_data, 
    save_data
)


__all__ = ['setup_logger', 'load_data', 'save_data']
try:
    from src.utils.pg_connect import PostgresHandler
    __all__.append('PostgresHandler')
except ImportError:
    # psycopg2 не установлен, но это не должно мешать использовать другие утилиты
    pass