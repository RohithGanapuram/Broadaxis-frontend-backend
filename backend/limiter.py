# limiter.py
import asyncio
import time

class MinuteBudgetLimiter:
    def __init__(self, rpm: int = 60, max_in_flight: int = 6):
        self.rpm = max(1, rpm)
        self.per_sec = self.rpm / 60.0
        self._tokens = float(self.rpm)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()
        self._in_flight = asyncio.Semaphore(max_in_flight)

    async def reserve(self):
        await self._in_flight.acquire()
        # simple leaky-bucket on request count
        while True:
            async with self._lock:
                now = time.monotonic()
                delta = now - self._last
                self._last = now
                self._tokens = min(self.rpm, self._tokens + delta * self.per_sec)
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                needed = (1.0 - self._tokens) / self.per_sec
            await asyncio.sleep(needed)

    def release(self):
        self._in_flight.release()
