"""Вежливый HTTP-клиент: пауза между запросами и повторы с экспоненциальной задержкой."""

import time
from dataclasses import dataclass

import httpx

USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/141.0.0.0 Safari/537.36")


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 4
    base_delay: float = 5.0

    def backoff(self, attempt: int) -> float:
        """Задержка перед повтором номер `attempt` (с нуля): base · 2^attempt."""
        return self.base_delay * 2 ** attempt


DEFAULT_RETRY = RetryPolicy()


def is_retryable(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


class PoliteFetcher:
    """GET-запросы не чаще одного в `min_interval` секунд; временные сбои повторяются."""

    def __init__(self, min_interval: float = 1.0, retry: RetryPolicy = DEFAULT_RETRY, timeout: float = 30):
        self._client = httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=timeout)
        self._min_interval = min_interval
        self._retry = retry
        self._last_request = 0.0

    def get(self, url: str, **params) -> httpx.Response:
        for attempt in range(self._retry.attempts):
            self._wait_turn()
            try:
                response = self._client.get(url, params=params or None)
                if is_retryable(response.status_code):
                    raise httpx.HTTPStatusError(str(response.status_code), request=response.request, response=response)
                response.raise_for_status()
                return response
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                if attempt == self._retry.attempts - 1:
                    raise
                pause = self._retry.backoff(attempt)
                print(f"  {url}: {e}, повтор через {pause:.0f} с", flush=True)
                time.sleep(pause)
        raise AssertionError("unreachable")

    def _wait_turn(self) -> None:
        wait = self._min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()
