import numpy as np
import pandas as pd
import multiprocessing as mp


def hypotenuse(row):
    return round(row[1]**2 + row[2]**2)**0.5


def sum_of_square(column):
    return sum([i**2 for i in column[1]])


if __name__ == "__main__":
    np.random.RandomState(seed=2018)
    df = pd.DataFrame(np.random.randint(3, 10, size=[5, 2]))
    print("data overview: " + "\n", df.head(5))

    with mp.Pool(processes=3) as pool:
        result_1 = pool.imap(hypotenuse, df.itertuples(name=False), chunksize=10)
        output_1 = [round(x, 2) for x in result_1]
        print(output_1)

        result_2 = pool.imap(sum_of_square, df.iteritems(), chunksize=10)
        output_2 = [x for x in result_2]
        print(output_2)

        df.iterrows()
