import ctypes
import concurrent.futures
import multiprocessing
import re
import torch.multiprocessing
import traceback

from torch.multiprocessing import current_process

is_main_process = not bool(re.match(r'((.*Process)|(SyncManager)|(.*PoolWorker))-\d+', current_process().name))
_interrupted: multiprocessing.Value


def main_process_print(self, *args, sep=' ', end='\n', file=None):
    if is_main_process:
        print(self, *args, sep=sep, end=end, file=file)


def multiprocess_worker_main(map_func, *args):
    global _interrupted
    if _interrupted.value:
        return None
    # noinspection PyBroadException
    try:
        return map_func(*args)
    except KeyboardInterrupt:
        _interrupted.value = True
        return None
    except Exception:
        traceback.print_exc()
        return None


def setup_interrupt_flag(sval):
    global _interrupted
    _interrupted = sval


def multiprocess_run(map_func, args, num_workers):
    num_jobs = len(args)
    if num_jobs < num_workers:
        num_workers = num_jobs

    sval = multiprocessing.Value(ctypes.c_bool, False)
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=torch.multiprocessing.get_context('spawn'),
            initializer=setup_interrupt_flag, initargs=(sval,)
    ) as executor:
        futures = [executor.submit(multiprocess_worker_main, map_func, *a) for a in args]
        try:
            yield from (f.result() for f in concurrent.futures.as_completed(futures))
        except KeyboardInterrupt:
            for f in futures:
                if not f.done():
                    f.cancel()
            global _interrupted
            _interrupted.value = True
            executor.shutdown(wait=True)
            raise
