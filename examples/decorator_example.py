import time
from statistics import mean
import pickle

# Simple caching function, will not work for anything 
# advanced since the arguments to the function needs 
# to be pickleable
def cacher(f):
    cache = dict()
    def cached_function(*args, **kwargs):
        key = pickle.dumps((args, kwargs))
        if key in cache:
            return cache[key]
        else:
            value = f(*args, **kwargs)
            cache[key] = value
            return value
    return cached_function


#@cacher  #try enabling or disabling this and see what happens
def slow_square(x):
    time.sleep(0.2)
    return x**2


if __name__ == '__main__':
    execution_times = []
    x = 1729
    for i in range(10):
        t0 = time.time()
        y = slow_square(x)
        dt = time.time() - t0
        execution_times.append(dt)
    print("Mean time", mean(execution_times))