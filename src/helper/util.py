import time
import logging
import pandas as pd
from pathlib import Path
## solve import issues in other file

DIR = Path(__file__).parent.parent

# print(DIR)
# print(DIR.joinpath(Path(__file__).stem))


class Colors:
    # NOTE: color codes are not working with f-strings with "="
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    BLACK = "\033[30m"
    BROWN = "\033[33m"
    PURPLE = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[1;30m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    ENDC = '\033[m'



class Estimate:
    @staticmethod
    def timer(func):
        import time
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            wrapper.total_time += duration
            print(Colors.BLUE + f"Took {duration:.6f} sec" + Colors.ENDC)
            return result

        wrapper.total_time = 0
        return wrapper
    
