"""Sample this process's RSS, avoiding inherited ru_maxrss high-water marks."""
from threading import Event, Thread
import os

import psutil


class MemorySampler:
    def __init__(self, interval=0.01):
        try:
            self.process = psutil.Process()
        except psutil.NoSuchProcess:
            # Some containers virtualize getpid() without remounting /proc.
            # /proc/self still resolves to the current process in that mount.
            self.process = psutil.Process(int(os.readlink("/proc/self")))
        self.interval = interval
        self.peak = self.process.memory_info().rss
        self.stop = Event()
        self.thread = Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.wait(self.interval):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()

    @property
    def peak_mb(self):
        return self.peak / (1024 * 1024)
