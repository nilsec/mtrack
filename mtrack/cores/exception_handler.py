import tblib.pickling_support
tblib.pickling_support.install()

import sys
from multiprocessing import Pool

class ExceptionWrapper(object):
    def __init__(self, ee):
        self.ee = ee
        __,  __, self.tb = sys.exc_info()

    def re_raise(self):
        raise self.ee, None, self.tb

    @staticmethod
    def exception_handler(result):
        if isinstance(result, ExceptionWrapper):
            result.re_raise()
 


# example how to use ExceptionWrapper with pool.apply_async

def inverse(i):
    """will fail for i == 0"""
    try:
        print 1.0 / i
    except Exception as e:
        return ExceptionWrapper(e)

def exception_handler(result):
    if isinstance(result, ExceptionWrapper):
        result.re_raise()
        
def main():
    pool = Pool()
    args = [1,2,3,4,0]
    for j in args:
        pool.apply_async(inverse, (j,), callback=ExceptionWrapper.exception_handler)
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
