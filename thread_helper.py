import threading
import ctypes

def get_thread_id(thread):
    # Assumes that the thread is running
    if hasattr(thread, "_thread_id"):
        return thread._thread_id
    for id, t in threading._active.items():
        if t is thread:
            return id
    return None

def raise_exception(thread_id):
    if thread_id == None:
        print("Can't raise Exception for NoneType thread_id")
        return
    
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SystemExit))
    if res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
        print('Exception raise failure')
