import multiprocessing as mp
from functools import partial
import numpy as np
from pathos.multiprocessing import ProcessingPool


def work_sum_1(x):
    return x + 1


def work_sum(x, y):
    return x + y


if __name__ == "__main__":
    with mp.Pool(processes=3) as pool:
        # generate data
        np.random.RandomState(seed=2018)
        l = np.random.randint(3, 10, size=[1, 5]).tolist()[0]
        print("overview of l: " + "\n", l)

        result_1 = pool.map(work_sum_1, l)
        print("result using Pool.map(): ", result_1)

        """
        when there is another list and we want to calculate the corresponding sum, 
        we cannot finish the job for Pool.map() can only take one argument.
        """
        _l = np.random.randint(10, 20, size=[1, 5]).tolist()[0]
        print("overview of _l: ", _l)

        # IMPLEMENT 1: using partial via setting y = 1
        partial_work_sum = partial(work_sum, y=1)
        result_2 = pool.map(partial_work_sum, l)
        print("result using partial: ", result_2)

        # IMPLEMENT 2: using pathos
        result_3 = ProcessingPool(3).map(work_sum, l, _l)
        print("result using pathos: ", result_3)

        # IMPLEMENT 3: combining two argument into one
        x_y = list(zip(l, _l))
        print("overview of x_y: ", x_y)
        result_4 = pool.starmap(work_sum, x_y)
        print("result using zip: ", result_4)
