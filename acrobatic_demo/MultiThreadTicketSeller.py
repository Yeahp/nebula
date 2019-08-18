import multiprocessing.dummy
import threading
import time
import os


def booth(tid):
    global i
    global lock
    while True:
        lock.acquire()
        if i != 0:
            i -= 1
            print("窗口: ", tid, ", 剩余票数: ", i)
            print("number of active threads: ", threading.active_count())
            print("the current thread: ", threading.current_thread())
            time.sleep(1)
        else:
            print("Thread_id: ", tid, " No more tickets")
            os._exit(0)
        lock.release()
        time.sleep(1)


if __name__ == "__main__":
    i = 30
    lock = threading.Lock()

    for k in range(5):
        new_thread = threading.Thread(target=booth, args=(k,))
        new_thread.start()
