from pydantic_settings import BaseSettings
from typing import List, Optional

# Класс для хранения переменных среды проекта
# Определены переменные, ожидаемые типы и дефолтные значения
class Settings(BaseSettings):
    DB_NAME: str = 'recsys'
    DB_USER: str = 'postgres'
    DB_PASSWORD: str = 'password'
    DB_HOST: str = 'localhost'
    DB_PORT: int = 5432

    class Config:
        env_file = ".env"


configs = Settings()