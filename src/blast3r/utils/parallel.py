# Copyright (C) 2026-present Naver Corporation. All rights reserved.
#
# Adapted for BLASt3R from DUSt3R (https://github.com/naver/dust3r),
# dust3r/utils/parallel.py.

from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count


def parallel_threads(function, args, workers=0, star=False, starstar=False, front_num=0, Pool=ThreadPool, **tqdm_kw):
    """A parallel version of map() with a progress bar.

    Args:
        args: an array to iterate over.
        function: a function to apply to the elements of args.
        workers: number of cores to use; all of them by default.
        star: whether to call function(*args).
        starstar: whether to call function(**args).
        front_num: number of iterations to run serially before starting the
            parallel job, which is useful for catching bugs.

    Returns:
        [function(args[0]), function(args[1]), ...]
    """
    try:
        total = len(args)
    except TypeError:
        args = list(args)
        total = len(args)

    with tqdm(total=total, **tqdm_kw) as progress_bar:
        while workers <= 0:
            workers += cpu_count()

        if workers == 1:
            front_num = total

        # we run the first few iterations serially to catch bugs
        front = []
        if front_num > 0:
            args = list(args) # explicitly list them
            for a in args[:front_num]:
                front.append(function(*a) if star else function(**a) if starstar else function(a))
                progress_bar.update(1)
            args = args[front_num:]

        # assemble the workers
        out = []
        with Pool(workers) as pool:
            # pass the elements of args into function
            if star:
                futures = pool.imap(starcall, [(function,a) for a in args])
            elif starstar:
                futures = pool.imap(starstarcall, [(function,a) for a in args])
            else:
                futures = pool.imap(function, args)
            # print out the progress as tasks complete
            for f in futures:
                out.append(f)
                progress_bar.update(1)
        return front + out


def starcall(args):
    """Convenience wrapper for Process.Pool."""
    function, args = args
    return function(*args)

def starstarcall(args):
    """Convenience wrapper for Process.Pool."""
    function, args = args
    return function(**args)
