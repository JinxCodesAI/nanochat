"""
Async wrapper for the dataloader generator.

Runs the underlying generator in a single producer thread, putting each
yielded item into a bounded FIFO queue. The consumer's `next()` just
dequeues, so parquet read + tokenization + best-fit + HtoD overlap with
the GPU's forward/backward instead of blocking the main thread.

Each queue entry is a (producer_dt_ms, item) tuple so a downstream timer
can attribute producer timings to the step the consumer actually consumed
them in (avoids off-by-one when producer is ahead of consumer).
"""

import queue
import threading
import time


_SENTINEL = object()


class async_loader:
    """Async iterator backed by a single producer thread + bounded queue.

    Args:
        generator: any iterable yielding the items the consumer expects.
            Must be infinite (or the consumer breaks when it gets the sentinel).
        maxsize: queue capacity. 2 keeps one batch "in flight" + one ready.

    StopIteration / exceptions propagate to the consumer via a sentinel +
    exception capture. Ordering is preserved (single producer, FIFO queue).
    """

    def __init__(self, generator, maxsize=2):
        self._q = queue.Queue(maxsize=maxsize)
        self._exc = []
        self._lock = threading.Lock()

        def _run():
            try:
                while True:
                    t0 = time.perf_counter()
                    item = next(generator)
                    dt = (time.perf_counter() - t0) * 1000.0
                    self._q.put((dt, item))
            except StopIteration:
                pass
            except BaseException as e:
                with self._lock:
                    self._exc.append(e)
            finally:
                self._q.put(_SENTINEL)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    @property
    def queue(self):
        return self._q

    def __iter__(self):
        return self

    def __next__(self):
        payload = self._q.get()
        if payload is _SENTINEL:
            with self._lock:
                if self._exc:
                    raise self._exc[0]
            raise StopIteration
        return payload
