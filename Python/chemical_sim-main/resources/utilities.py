import os
import time


def optimization_result_file_path(file_stem: str, timestamp: str) -> str:
    return os.path.join("results", f"{file_stem}-{timestamp}.txt")


def run_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(time.time() - start_time)
        return result
    return wrapper
