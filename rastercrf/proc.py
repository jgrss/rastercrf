import itertools
import concurrent.futures


class Concurrency(object):

    def __init__(self, func, tasks, chunk_size=100, max_workers=1):

        self.func = func
        self.tasks = tasks
        self.chunk_size = chunk_size
        self.max_workers = max_workers

    def exec(self, X, y):

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:

            futures = {executor.submit(self.func, *taskargs) for taskargs in itertools.islice(self.tasks, self.chunk_size)}

            while futures:

                done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

                for f in done:

                    X_, y_ = f.result()

                    X.append(X_)
                    y.append(y_)

                for taskargs in itertools.islice(self.tasks, len(done)):
                    futures.add(executor.submit(self.func, *taskargs))

        return X, y
