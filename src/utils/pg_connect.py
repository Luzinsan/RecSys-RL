import psycopg2
from psycopg2 import sql
from psycopg2.extensions import connection as pg_connection
from typing import Optional, Tuple
from time import time
import pandas as pd
import logging
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from src.config.configs import settings


class PostgresHandler:


    def __init__(self, settings):
        self.pg_conn: Optional[pg_connection] = None
        self.sqlalchemy_engine: Optional[Engine] = None
        self.config = dict(
            dbname=settings.DB_NAME,
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD
        )
        self.connection_string = f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"


    def __enter__(self):
        """
        Creates a connection to the database when entering the context manager
        """
        self.pg_conn = psycopg2.connect(**self.config)
        self.sqlalchemy_engine = create_engine(self.connection_string)
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Closes the database connection when exiting the context manager
        """
        if self.pg_conn:
            self.pg_conn.close()
        if self.sqlalchemy_engine:
            self.sqlalchemy_engine.dispose()


    def post(self, query: str) -> float:
        """
        Выполняет SQL-запрос на отправку данных в базу данных и возвращает время выполнения
        """
        start_time = time()
        try:
            with self.pg_conn.cursor() as cursor:
                cursor.execute(sql.SQL(query))
                self.pg_conn.commit()
                return time() - start_time
        except (Exception, psycopg2.Error) as error:
            logging.error(f"Ошибка при отправке данных в PostgreSQL: {error}")
            return 0.0


    def get(self, query: str) -> Tuple[pd.DataFrame, float]:
        """
        Выполняет SQL-запрос на получение данных из базы данных и возвращает результат в виде DataFrame и время выполнения
        """
        start_time = time()
        try:
            df = pd.read_sql(query, self.sqlalchemy_engine)
            return df, time() - start_time
        except Exception as error:
            logging.error(f"Ошибка при запросе данных из PostgreSQL: {error}")
            return pd.DataFrame(), 0.0


    @staticmethod
    def send(query: str) -> pd.DataFrame:
        """
        Connects to the database and executes an SQL query to read data and returns the result as a DataFrame.
        
        Args:
            query (str): SQL query to execute
            
        Returns:
            pd.DataFrame: Query result as a DataFrame
            
        Raises:
            Exception: If an error occurs during query execution
        """
        logging.info(f"Executing SQL query: {query[:100]}...")
        try:
            with PostgresHandler(settings) as handler:
                result, execution_time = handler.get(query)
                logging.info(f"Query executed successfully in {execution_time:.2f} sec. Data size: {result.shape}")
                return result
        except Exception as e:
            logging.error(f"Error executing query: {str(e)}")
            raise
