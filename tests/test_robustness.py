import importlib.util
import sys

import httpx
import pytest

from gigarag.atomic import atomic_save, temp_path
from gigarag.cache import CacheStore, code_fingerprint, load_or_compute
from gigarag.http import PoliteFetcher, RetryPolicy, send_with_retries
from gigarag.sources.dota2ru import pending_comment_pages
from gigarag.sources.topic_lexicon import build_topic_matcher

NO_WAIT = RetryPolicy(attempts=3, base_delay=0)


# ---------- повторы HTTP ----------

def fetcher_with(statuses: list[int | Exception]) -> tuple[PoliteFetcher, list[int]]:
    """PoliteFetcher поверх фейкового транспорта, отвечающего по очереди заданными статусами."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        outcome = statuses[min(len(calls), len(statuses) - 1)]
        calls.append(1)
        if isinstance(outcome, Exception):
            raise outcome
        return httpx.Response(outcome, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return PoliteFetcher(min_interval=0, retry=NO_WAIT, client=client), calls


def test_permanent_4xx_is_not_retried():
    fetcher, calls = fetcher_with([404])
    with pytest.raises(httpx.HTTPStatusError):
        fetcher.get("https://example.test/x")
    assert len(calls) == 1


def test_temporary_errors_are_retried_until_success():
    fetcher, calls = fetcher_with([503, httpx.ConnectError("boom"), 200])
    assert fetcher.get("https://example.test/x").status_code == 200
    assert len(calls) == 3


def test_retries_are_limited():
    fetcher, calls = fetcher_with([429])
    with pytest.raises(httpx.HTTPStatusError):
        fetcher.get("https://example.test/x")
    assert len(calls) == NO_WAIT.attempts


def test_send_with_retries_sleeps_with_backoff():
    pauses: list[float] = []
    request = httpx.Request("GET", "https://example.test")
    responses = iter([httpx.Response(500, request=request), httpx.Response(200, request=request)])
    send_with_retries(lambda: next(responses), RetryPolicy(attempts=3, base_delay=2), sleep=pauses.append,
                      log=lambda _: None)
    assert pauses == [2]


def test_exhausted_quota_is_not_retried():
    calls: list[int] = []
    request = httpx.Request("POST", "https://example.test")
    body = {"error": {"type": "insufficient_quota", "message": "You have no credits remaining."}}

    def send() -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, json=body, request=request)
    with pytest.raises(httpx.HTTPStatusError):
        send_with_retries(send, RetryPolicy(attempts=3, base_delay=0), sleep=lambda _: None, log=lambda _: None)
    assert len(calls) == 1


# ---------- атомарная запись ----------

def test_interrupted_save_leaves_no_file(tmp_path):
    target = tmp_path / "emb.npy"

    def broken_save(value, path):
        path.write_text("half")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        atomic_save(broken_save, "value", target)
    assert not target.exists() and not temp_path(target).exists()


def test_load_or_compute_recomputes_after_interrupted_save(tmp_path):
    path = tmp_path / "x.txt"
    load = lambda p: p.read_text()  # noqa: E731

    def failing_save(v, p):
        raise RuntimeError

    with pytest.raises(RuntimeError):
        load_or_compute(path, lambda: "v1", load=load, save=failing_save)
    assert load_or_compute(path, lambda: "v2", load=load, save=lambda v, p: p.write_text(v)) == "v2"


def test_temp_path_keeps_extension(tmp_path):
    assert temp_path(tmp_path / "a.npy").suffix == ".npy"


# ---------- отпечаток кода в ключе кэша ----------

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_code_change_changes_cache_path(tmp_path):
    (tmp_path / "algo_v1.py").write_text("def f(x):\n    return x + 1\n")
    (tmp_path / "algo_v2.py").write_text("def f(x):\n    return x + 2\n")
    v1, v2 = load_module(tmp_path / "algo_v1.py", "algo_v1"), load_module(tmp_path / "algo_v2.py", "algo_v2")
    store = CacheStore(tmp_path / "cache", "corpus-sha")
    assert code_fingerprint(v1) != code_fingerprint(v2)
    assert store.path("emb", "npy", code=code_fingerprint(v1)) != store.path("emb", "npy", code=code_fingerprint(v2))


# ---------- тематический словарь ----------

@pytest.mark.parametrize("text", ["следующий этап турнира", "новый сетап и кейкап", "результаты матча",
                                  "описание предмета скрыто"[:12]])
def test_topic_terms_do_not_match_inside_words(text):
    assert not build_topic_matcher([])(text)


@pytest.mark.parametrize("text", ["пуджа апнули", "ап лины", "понерфили войда", "ульта стала сильнее",
                                  "новая мета", "в патче 7.41"])
def test_topic_terms_match_word_starts(text):
    assert build_topic_matcher([])(text)


# ---------- докачка страниц комментариев ----------

def test_pending_comment_pages():
    html = 'href="/news/1-a?page=2" href="/news/1-a?page=3" href="/news/1-a?page=1"'
    assert pending_comment_pages(html, "/news/1-a", downloaded=lambda n: n == 2) == [3]
    assert pending_comment_pages(html, "/news/1-a", downloaded=lambda n: True) == []
