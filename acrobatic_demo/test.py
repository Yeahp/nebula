'''
import threading
from time import sleep


def test(n, event):
    #while not event.isSet():
    #    print('Thread %s is ready' % n)
    #    sleep(1)
    print('Thread %s is ready' % n)
    event.wait()
    while event.isSet():
        print('Thread %s is running' % n)
        sleep(1)


def main():
    event = threading.Event()
    #for i in range(0, 2):
    #    th = threading.Thread(target=test, args=(i, event))
    #    th.start()
    th1 = threading.Thread(target=test, args=(0, event))
    th2 = threading.Thread(target=test, args=(1, event))
    th1.start()
    th2.start()
    sleep(3)
    print('----- event is set -----')
    event.set()
    sleep(3)
    print('----- event is clear -----')
    event.clear()
    print(th1.is_alive())
    print(th1.is_alive())


if __name__ == '__main__':
    main()
'''

import threading
import time

class PeriodicTimer:
    def __init__(self, interval):
        self._interval = interval
        self._flag = 0
        self._cv = threading.Condition()

    def start(self):
        t = threading.Thread(target=self.run)
        t.daemon = True

        t.start()

    def run(self):
        '''
        Run the timer and notify waiting threads after each interval
        '''
        while True:
            time.sleep(self._interval)
            with self._cv:
                 self._flag ^= 1
                 self._cv.notify_all()

    def wait_for_tick(self):
        '''
        Wait for the next tick of the timer
        '''
        with self._cv:
            last_flag = self._flag
            while last_flag == self._flag:
                self._cv.wait()

# Example use of the timer
ptimer = PeriodicTimer(5)
ptimer.start()

# Two threads that synchronize on the timer
def countdown(nticks):
    while nticks > 0:
        ptimer.wait_for_tick()
        print('T-minus', nticks)
        nticks -= 1

def countup(last):
    n = 0
    while n < last:
        ptimer.wait_for_tick()
        print('Counting', n)
        n += 1

threading.Thread(target=countdown, args=(10,)).start()
threading.Thread(target=countup, args=(5,)).start()