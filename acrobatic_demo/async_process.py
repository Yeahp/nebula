import numpy as np
import multiprocessing as mp


def howmany_within_range(row, minimum, maximum):
    """
    :param row:
    :param minimum:
    :param maximum:
    :return: how many numbers lie within `minimum` and `maximum` in a given `row`
    """
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count += 1
    return count


def howmany_within_range_rowonly(row, minimum=4, maximum=8):
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count += 1
    return count


def howmany_within_range2(i, row, minimum, maximum):
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count += 1
    return (i, count)


def collect_result(result):
    global result_5
    result_5.append(result)


if __name__ == "__main__":
    # show how many cpus in this computer
    print("number of processors: ", mp.cpu_count())

    # QUESTION: calculate the number of numbers in a given range
    # prepare data
    np.random.RandomState(2018)
    data = np.random.randint(0, 10, size=[200000, 5]).tolist()

    # METHOD-1: without parallellization
    result_1 = []
    for row in data:
        result_1.append(howmany_within_range(row, 4, 8))
    print("result without parallellization: ", result_1[:5])

    # METHOD-2: with parallellizing
    pool = mp.Pool(mp.cpu_count())
    # 2.1 - using Pool.apply()
    result_2 = [pool.apply(howmany_within_range, args=(row, 4, 8)) for row in data]
    print("result using Pool.apply(): ", result_2[:5])

    # 2.2 - using Pool.map()
    result_3 = pool.map(howmany_within_range_rowonly, [row for row in data])
    print(result_3[:5])

    # 2.3 - using Pool.starmap()
    result_4 = pool.starmap(howmany_within_range, [(row, 4, 8) for row in data])
    print("result using Pool.starmap(): ", result_4[:5])

    # 2.4 - using Pool.apply_async()
    result_5 = []
    for i, row in enumerate(data):
        pool.apply_async(howmany_within_range2, args=(i, row, 4, 8), callback=collect_result)
    result_5.sort(key=lambda x: x[0])
    print("result using Pool.apply_async(): ", [r for i, r in result_5[:5]])

    _result_5 = [pool.apply_async(howmany_within_range2, args=(i, row, 4, 8)) for i, row in enumerate(data)]
    _result_5_get = [r.get() for r in _result_5]
    _result_5_get.sort(key=lambda x: x[0])
    print("result using Pool.apply_async(): ", [r for i, r in _result_5_get[:5]])

    # 2.5 - using Pool.starmap_async()
    result_6 = pool.starmap_async(howmany_within_range2, [(i, row, 4, 8) for i, row in enumerate(data)]).get()
    result_6.sort(key=lambda x: x[0])
    print("result using Pool.starmap_async(): ", [r for i, r in result_6[:5]])

    pool.close()


