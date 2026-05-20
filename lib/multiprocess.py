import platform
import traceback

from torch.multiprocessing import Manager, Process, get_context


class FailedItem:
    """Sentinel placed on the queue when a worker fails to process an item."""
    __slots__ = ("exception", "traceback_str")

    def __init__(self, exception: Exception, tb: str):
        self.exception = repr(exception)
        self.traceback_str = tb


def chunked_worker_run(map_func, args, results_queue=None):
    for a in args:
        try:
            res = map_func(*a)
            results_queue.put(res)
        except KeyboardInterrupt:
            break
        except Exception as exc:
            results_queue.put(FailedItem(exc, traceback.format_exc()))


def chunked_multiprocess_run(map_func, args, num_workers, q_max_size=1000):
    num_jobs = len(args)
    if num_jobs < num_workers:
        num_workers = num_jobs

    queues = [Manager().Queue(maxsize=q_max_size // num_workers) for _ in range(num_workers)]
    if platform.system().lower() != 'windows':
        process_creation_func = get_context('spawn').Process
    else:
        process_creation_func = Process

    workers = []
    for i in range(num_workers):
        worker = process_creation_func(
            target=chunked_worker_run, args=(map_func, args[i::num_workers], queues[i]), daemon=True
        )
        workers.append(worker)
        worker.start()

    for i in range(num_jobs):
        yield queues[i % num_workers].get()

    for worker in workers:
        worker.join()
        worker.close()
