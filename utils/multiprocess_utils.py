import platform
import re
import traceback

from torch.multiprocessing import Manager, Process, current_process, get_context

is_main_process = not bool(re.match(r'((.*Process)|(SyncManager)|(.*PoolWorker))-\d+', current_process().name))


def main_process_print(self, *args, sep=' ', end='\n', file=None):
    if is_main_process:
        print(self, *args, sep=sep, end=end, file=file)


def chunked_worker_run(map_func, args, results_queue=None):
    for a in args:
        # noinspection PyBroadException
        try:
            res = map_func(*a)
            results_queue.put(res)
        except KeyboardInterrupt:
            break
        except Exception:
            traceback.print_exc()
            results_queue.put(None)


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
