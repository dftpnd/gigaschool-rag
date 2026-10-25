"""Общее для сборщиков: HTTP-клиент с паузой и пути к сырым данным."""

import time
from pathlib import Path

import httpx

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"


class Fetcher:
    """httpx-клиент, который выдерживает паузу между запросами и повторяет при сбоях."""

    def __init__(self, delay: float = 1.0, retries: int = 4):
        self.client = httpx.Client(headers={"User-Agent": UA}, follow_redirects=True, timeout=30)
        self.delay = delay
        self.retries = retries
        self._last = 0.0

    def get(self, url: str, **params) -> httpx.Response:
        for attempt in range(self.retries):
            wait = self.delay - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.client.get(url, params=params or None)
                if r.status_code == 429 or r.status_code >= 500:
                    raise httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
                r.raise_for_status()
                return r
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                if attempt == self.retries - 1:
                    raise
                pause = 5 * 2**attempt
                print(f"  {url}: {e}, повтор через {pause} с", flush=True)
                time.sleep(pause)
        raise AssertionError("unreachable")
