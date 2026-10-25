"""Вежливый HTTP-клиент: пауза между запросами и повторы временных сбоев с экспоненциальной задержкой."""

import time
from collections.abc import Callable
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
    """Временный сбой сервера: 429 и 5xx. Остальные 4xx повторять бессмысленно."""
    return status_code == 429 or status_code >= 500


def is_quota_exhausted(response: httpx.Response) -> bool:
    """429 из-за кончившегося баланса API (`insufficient_quota`), а не из-за частоты запросов: повтор не поможет."""
    try:
        return response.json().get("error", {}).get("type") == "insufficient_quota"
    except ValueError:
        return False


def send_with_retries(send: Callable[[], httpx.Response], retry: RetryPolicy,
                      sleep: Callable[[float], None] = time.sleep,
                      log: Callable[[str], None] = print) -> httpx.Response:
    """Выполняет `send()`, повторяя только сетевые ошибки и временные статусы (`is_retryable`).
    Постоянные ошибки (404, 401, …) пробрасываются сразу как `HTTPStatusError`."""
    for attempt in range(retry.attempts):
        last = attempt == retry.attempts - 1
        try:
            response = send()
        except httpx.TransportError as e:
            if last:
                raise
            reason = repr(e)
        else:
            if not is_retryable(response.status_code) or is_quota_exhausted(response) or last:
                return response.raise_for_status()
            reason = f"HTTP {response.status_code}"
        pause = retry.backoff(attempt)
        log(f"  {reason}, повтор через {pause:.0f} с")
        sleep(pause)
    raise AssertionError("unreachable")


class PoliteFetcher:
    """GET-запросы не чаще одного в `min_interval` секунд; временные сбои повторяются."""

    def __init__(self, min_interval: float = 1.0, retry: RetryPolicy = DEFAULT_RETRY, timeout: float = 30,
                 client: httpx.Client | None = None):
        self._client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True,
                                              timeout=timeout)
        self._min_interval = min_interval
        self._retry = retry
        self._last_request = 0.0

    def get(self, url: str, **params) -> httpx.Response:
        def send() -> httpx.Response:
            self._wait_turn()
            return self._client.get(url, params=params or None)
        return send_with_retries(send, self._retry, log=lambda m: print(f"  {url}:{m}", flush=True))

    def _wait_turn(self) -> None:
        wait = self._min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()
