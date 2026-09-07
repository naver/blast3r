# Copyright (C) 2026-present Naver Corporation. All rights reserved.

import time
from collections import defaultdict
import numpy as np
from contextlib import contextmanager

import threading
import torch

class TimerEntry:
    def __init__(self):
        self.splits = []

        self.start_time = None

    def start(self):
        assert self.start_time is None, f"Timer {self.name} has already been started."

        self.start_time = time.time()

    def stop(self):
        assert self.start_time is not None, f"Timer {self.name} has not been started."

        end_time = time.time()
        self.splits.append(end_time - self.start_time)
        self.start_time = None

    def summary(self):
        arr = np.array(self.splits)
        total_time = np.sum(arr)
        avg = np.mean(arr)
        std = np.std(arr)
        count = len(self.splits)
        return {
            'total_time': total_time,
            'count': count,
            'avg': avg,
            'std': std,
            'splits': self.splits,
        }

class Timer:
    """A timer utility for measuring execution time of code blocks.

    This class provides two ways to time code execution:
    1. Manual start/stop method calls
    2. Context manager (with statement)

    The timer maintains statistics for multiple named operations and can
    provide summaries including total time, average time, standard deviation,
    and execution count for each named operation.

    Examples:
        Manual timing:
            timer = Timer()
            timer.start('data_loading')
            # ... load data ...
            timer.stop('data_loading')

            timer.start('processing')
            # ... process data ...
            timer.stop('processing')

            print(timer.summary())

        Context manager timing:
            timer = Timer()
            with timer.time('data_loading'):
                # ... load data ...
                pass

            with timer.time('processing'):
                # ... process data ...
                pass

            print(timer.summary())

        Mixed usage:
            timer = Timer()
            timer.start('setup')
            # ... setup code ...
            timer.stop('setup')

            for i in range(10):
                with timer.time('iteration'):
                    # ... iteration code ...
                    pass

            # Get statistics for all timed operations
            stats = timer.summary()
            print(f"Setup time: {stats['setup']['total_time']:.3f}s")
            print(f"Avg iteration: {stats['iteration']['avg']:.3f}s")
    """

    def __init__(self):
        self.timers = defaultdict(TimerEntry)

    def start(self, name):
        self.timers[name].start()

    def stop(self, name):
        self.timers[name].stop()

    def summary(self):
        return {name: timer.summary() for name, timer in self.timers.items()}

    def reset(self):
        self.timers = defaultdict(TimerEntry)

    @contextmanager
    def time(self, name):
        """Context manager for timing code blocks.

        Usage:
            with timer.time('my_operation'):
                # code to time
                pass
        """
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)


timer = Timer()


MB = 1024*1024
class GPUMonitor:
    """Monitors GPU memory and utilization periodically. Global, not named."""
    def __init__(self, device=None, interval=2.0):
        self.device = device if device is not None else torch.cuda.current_device()
        self.interval = interval
        self.memory_allocated = []
        self.memory_reserved = []
        self.memory_allocated_max = []
        self.memory_reserved_max = []
        self.timestamps = []
        self._thread = None
        self._stop_event = threading.Event()

    def _collect(self):
        while not self._stop_event.is_set():
            mem_alloc = torch.cuda.memory_allocated(self.device) / MB
            mem_res = torch.cuda.memory_reserved(self.device) / MB
            mem_alloc_max = torch.cuda.max_memory_allocated(self.device) / MB
            mem_res_max = torch.cuda.max_memory_reserved(self.device) / MB
            timestamp = time.time()
            self.memory_allocated.append(mem_alloc)
            self.memory_reserved.append(mem_res)
            self.memory_allocated_max.append(mem_alloc_max)
            self.memory_reserved_max.append(mem_res_max)
            self.timestamps.append(timestamp)
            self._stop_event.wait(self.interval)

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("GPUMonitor already running.")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collect, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._thread = None

    def summary(self):
        def stats(values):
            arr = np.array(values)
            return {
                'avg': float(np.mean(arr)) if arr.size else 0.0,
                'max': float(np.max(arr)) if arr.size else 0.0,
                'min': float(np.min(arr)) if arr.size else 0.0,
                'std': float(np.std(arr)) if arr.size else 0.0,
                'count': int(arr.size),
                'values': values
            }
        return {
            'memory_allocated': stats(self.memory_allocated),
            'memory_reserved': stats(self.memory_reserved),
            'memory_allocated_max': stats(self.memory_allocated_max),
            'memory_reserved_max': stats(self.memory_reserved_max),
            'timestamps': self.timestamps
        }

    def reset(self):
        self.memory_allocated = []
        self.memory_reserved = []
        self.memory_allocated_max = []
        self.memory_reserved_max = []
        self.timestamps = []

    @contextmanager
    def monitor(self):
        self.start()
        try:
            yield
        finally:
            self.stop()

