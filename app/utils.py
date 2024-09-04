import time
from typing import Callable
from .logger import main_logger

def log_time(func: Callable):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        main_logger.info(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper