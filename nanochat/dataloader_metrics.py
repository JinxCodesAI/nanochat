"""
Transparent dataloader timing wrapper for measuring whether the GPU is waiting
on data during pretraining.

Motivation
----------
The training loop in `scripts/base_train.py` calls `next(train_loader)` between
backward and the next forward pass inside the gradient-accumulation loop.
If the dataloader (tokenization + best-fit packing + HtoD copy) takes longer
than the GPU's forward+backward compute, the GPU stalls waiting for data.

This module provides a thin wrapper that times every `next()` call so we can
measure this directly instead of guessing from FLOPs/MFU arithmetic.

How to interpret the metrics
----------------------------
`data_wait_ms_mean` is the mean wall-clock time of `next(train_loader)` calls
within a step (one per micro-step).

`data_wait_ms_max` is the worst `next()` time within the step — useful for
catching tail latency spikes (GC pauses, parquet boundary effects, etc.).

`data_wait_pct_of_step` is `sum(next_times) / step_dt * 100`. This is the
fraction of wall-clock step time spent inside `next()`.

How to detect GPU starvation
----------------------------
Compare `data_wait_ms_mean` to GPU compute time per micro-step (visible via
the `bf16_mfu` log line and step dt). If `data_wait_ms_mean` >= GPU compute
time per micro-step, the dataloader is the bottleneck and the GPU idles for
the difference. If `data_wait_ms_mean` < GPU compute time, the dataloader is
hidden behind GPU compute (no penalty).

Usage
-----
    from nanochat.dataloader_metrics import DataloaderTimer

    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(...)
    train_loader = DataloaderTimer(train_loader)

    # In the training loop, use as a normal iterator:
    x, y, state = next(train_loader)

    # After each step, call end_step() to roll up stats:
    train_loader.end_step()
    stats = train_loader.summary()  # returns dict of latest step + cumulative

The wrapper is designed to be a drop-in replacement for the bare loader and
adds only a `time.perf_counter()` call around `next()` — overhead is
sub-microsecond and not measurable in practice.
"""

import time
from statistics import mean


class DataloaderTimer:
    """Thin transparent wrapper that times every next() call.

    Maintains two sets of stats:
      - Per-step stats: reset every step via `end_step()`. These power the
        log line and wandb metric for the current step.
      - Cumulative stats: running totals across the whole run. Useful for
        long-run averages.
    """

    def __init__(self, loader):
        self.loader = loader
        # Per-step accumulators (reset by end_step)
        self._step_times_ms = []
        # Per-step summary (populated by end_step)
        self.last_step_mean_ms = 0.0
        self.last_step_max_ms = 0.0
        self.last_step_count = 0
        self.last_step_total_ms = 0.0
        # Cumulative
        self.total_calls = 0
        self.total_time_ms = 0.0
        self.max_time_ms = 0.0

    def __iter__(self):
        return self

    def __next__(self):
        t0 = time.perf_counter()
        item = next(self.loader)
        t1 = time.perf_counter()
        dt_ms = (t1 - t0) * 1000.0
        self._step_times_ms.append(dt_ms)
        self.total_calls += 1
        self.total_time_ms += dt_ms
        if dt_ms > self.max_time_ms:
            self.max_time_ms = dt_ms
        return item

    def end_step(self):
        """Roll up the current step's per-next() times and reset.

        Call this once at the end of each training step, after the optimizer
        step and before/after the per-step logging.
        """
        if self._step_times_ms:
            self.last_step_mean_ms = mean(self._step_times_ms)
            self.last_step_max_ms = max(self._step_times_ms)
            self.last_step_count = len(self._step_times_ms)
            self.last_step_total_ms = sum(self._step_times_ms)
        else:
            self.last_step_mean_ms = 0.0
            self.last_step_max_ms = 0.0
            self.last_step_count = 0
            self.last_step_total_ms = 0.0
        self._step_times_ms = []

    def summary(self):
        """Return a dict of metrics for the last completed step + cumulative run."""
        return {
            "data_wait_ms_mean": self.last_step_mean_ms,
            "data_wait_ms_max": self.last_step_max_ms,
            "data_wait_calls": self.last_step_count,
            "data_wait_total_ms": self.last_step_total_ms,
            "data_wait_total_calls": self.total_calls,
            "data_wait_total_time_ms": self.total_time_ms,
            "data_wait_max_ms_ever": self.max_time_ms,
        }


class QueueDepthTimer:
    """Wraps an async_loader to measure consumer-side starvation.

    Semantics:
      queue_get_total_ms: total time spent inside Queue.get() this step (includes lock noise)
      queue_get_max_ms:   max single Queue.get() time this step (cleanest starvation signal)
      producer_total_ms:  total time producer spent per __next__ this step (dataloader work)
      producer_max_ms:    max single producer __next__ time this step
      starved_steps:      cumulative count of steps where queue_get_max_ms > STARVED_THRESHOLD_MS

    Drop-in replacement for DataloaderTimer in async-loader pipelines.

    Producer timings are pulled from the (producer_dt, item) tuple the
    async_loader puts on the queue, so they're exactly aligned with the
    items the consumer consumed this step (no off-by-one when the producer
    runs ahead of the consumer).
    """

    STARVED_THRESHOLD_MS = 1.0

    def __init__(self, async_iter):
        from nanochat.dataloader_async import async_loader as _async_loader
        self.loader = async_iter
        self._q = async_iter.queue
        self._step_get_total_ms = 0.0
        self._step_get_max_ms = 0.0
        self._step_producer_times_ms = []
        self._last_producer_total_ms = 0.0
        self._last_producer_max_ms = 0.0
        self._last_get_total_ms = 0.0
        self._last_get_max_ms = 0.0
        self.cum_starved_steps = 0
        self.cum_get_total_ms = 0.0
        self.cum_producer_total_ms = 0.0

    def __iter__(self):
        return self

    def __next__(self):
        from nanochat.dataloader_async import _SENTINEL
        t0 = time.perf_counter()
        payload = self._q.get()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._step_get_total_ms += dt_ms
        if dt_ms > self._step_get_max_ms:
            self._step_get_max_ms = dt_ms
        if payload is _SENTINEL:
            if self.loader._exc:
                raise self.loader._exc[0]
            raise StopIteration
        producer_dt_ms, item = payload
        self._step_producer_times_ms.append(producer_dt_ms)
        return item

    def end_step(self):
        # Snapshot current step's queue_get metrics BEFORE reset so summary() can read them.
        self._last_get_total_ms = self._step_get_total_ms
        self._last_get_max_ms = self._step_get_max_ms

        # Producer timings were appended in lockstep with consumer __next__ calls,
        # so this list corresponds exactly to the micro-batches consumed this step.
        producer_times = self._step_producer_times_ms
        if producer_times:
            self._last_producer_total_ms = sum(producer_times)
            self._last_producer_max_ms = max(producer_times)
        else:
            self._last_producer_total_ms = 0.0
            self._last_producer_max_ms = 0.0

        if self._last_get_max_ms > self.STARVED_THRESHOLD_MS:
            self.cum_starved_steps += 1

        self.cum_get_total_ms += self._last_get_total_ms
        self.cum_producer_total_ms += self._last_producer_total_ms

        self._step_get_total_ms = 0.0
        self._step_get_max_ms = 0.0
        self._step_producer_times_ms = []

    def summary(self):
        return {
            "queue_get_total_ms": self._last_get_total_ms,
            "queue_get_max_ms": self._last_get_max_ms,
            "queue_get_cum_total_ms": self.cum_get_total_ms,
            "producer_total_ms": self._last_producer_total_ms,
            "producer_max_ms": self._last_producer_max_ms,
            "producer_cum_total_ms": self.cum_producer_total_ms,
            "queue_starved_steps": self.cum_starved_steps,
            "queue_starved_threshold_ms": self.STARVED_THRESHOLD_MS,
        }
