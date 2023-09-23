import functools
import signal
import contextlib
import time

def timeout(sec):
    """
    timeout decorator
    :param sec: function raise TimeoutError after ? seconds
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):

            def _handle_timeout(signum, frame):
                err_msg = f'Function {func.__name__} timed out after {sec} seconds'
                raise TimeoutError(err_msg)

            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(sec)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapped_func
    return decorator

@contextlib.contextmanager
def timeout_context(msg=None, timeout_sec=1, log_func=print):
    def _handle_timeout(signum, frame):
        err_msg = f'{msg} timed out after {timeout_sec} seconds'
        raise TimeoutError(err_msg)

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(timeout_sec)
    yield
    signal.alarm(0)
    # time_elapsed = time.perf_counter() - begin_time
    # log_func(f"{msg or 'timer'} | {time_elapsed} sec")

@contextlib.contextmanager
def timer_context(msg=None, log_func=print):
    begin_time = time.perf_counter()
    yield
    time_elapsed = time.perf_counter() - begin_time
    log_func(f"{msg or 'timer'} | {time_elapsed} sec")

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        begin_time = time.perf_counter()
        res = func(*args, **kwargs)
        time_elapsed = time.perf_counter() - begin_time
        print(f'{func.__name__} | {time_elapsed} sec')
        return res
    return wrapper