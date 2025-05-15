import time
from functools import wraps
from typing import Callable, Any, Type
import sys
sys.path.append(r"C:\Users\Abdessamad\Desktop\MLOpsClassificationTexteV2\src")

import time

def log_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        print(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper