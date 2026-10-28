"""Bounded single-worker queue; cancellation does not release active work early."""

import asyncio
from concurrent.futures import ThreadPoolExecutor


class QueueBusy(Exception):
    pass


class WorkQueue:
    def __init__(self, capacity):
        self.queue = asyncio.Queue(maxsize=capacity)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-worker")
        self.closed = False
        self.active = False
        self.worker = asyncio.create_task(self.run())

    def submit(self, operation):
        if self.closed:
            raise QueueBusy()
        future = asyncio.get_running_loop().create_future()
        try:
            self.queue.put_nowait((operation, future))
        except asyncio.QueueFull:
            raise QueueBusy() from None
        future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return future

    async def run(self):
        while True:
            item = await self.queue.get()
            try:
                if item is None:
                    return
                operation, future = item
                self.active = True
                try:
                    result = await asyncio.get_running_loop().run_in_executor(self.executor, operation)
                except Exception as exc:
                    if not future.done():
                        future.set_exception(exc)
                else:
                    if not future.done():
                        future.set_result(result)
                finally:
                    self.active = False
            finally:
                self.queue.task_done()

    async def close(self):
        self.closed = True
        await self.queue.join()
        self.queue.put_nowait(None)
        await self.worker
        self.executor.shutdown(wait=True)
