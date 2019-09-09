import pandas as pd
import numpy as np
import time


def timeit(func):
    def wrapper(*args, **kwargs):
        print('start running function...')
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print('finish running function...')
        duration = (end - start) * 1000
        print(f'duration: {duration} ms')
    return wrapper


@timeit
def demo():
    time.sleep(10)


if __name__ == '__main__':
    demo()
