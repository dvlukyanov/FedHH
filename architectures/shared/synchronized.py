from functools import wraps
import threading


lock = threading.Lock()


def synchronized(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        with lock:
            return f(*args, **kwargs)
    return wrapped