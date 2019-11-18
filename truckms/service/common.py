import time
import threading


def time_regular(list_funcs, time_interval):
    while True:
        time.sleep(time_interval)
        for f in list_funcs:
            f()

def start_update_thread(list_funcs, time_interval):
    thread1 = threading.Thread(target=time_regular, args=(list_funcs, time_interval))
    thread1.start()
    return thread1