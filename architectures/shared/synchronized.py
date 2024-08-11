from functools import wraps
import threading


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


lock = threading.Lock()


def synchronized(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        with lock:
            return f(*args, **kwargs)
    return wrapped