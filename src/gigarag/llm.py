"""Вызовы LLM через OpenAI-совместимый Chat Completions API с кэшем ответов на диске.

Каждый запрос полностью описан `ChatRequest` (модель, сообщения, температура, seed, формат) —
его хэш служит ключом кэша, поэтому повторный прогон воспроизводит ответы и ничего не стоит.
"""

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from gigarag.atomic import atomic_write_text
from gigarag.http import DEFAULT_RETRY, RetryPolicy, send_with_retries


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: tuple[tuple[str, str], ...]  # (role, content)
    temperature: float
    seed: int | None = None
    json_mode: bool = False
    thinking: bool | None = None  # скрытое рассуждение модели; None — по умолчанию провайдера, поле не отправляется

    def key(self) -> str:
        fields = asdict(self)
        if self.thinking is None:
            del fields["thinking"]  # ключи запросов без явного режима — те же, что до появления поля (кэши КТ 1–3)
        return hashlib.sha256(json.dumps(fields, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def payload(self) -> dict:
        body = {"model": self.model, "temperature": self.temperature,
                "messages": [{"role": r, "content": c} for r, c in self.messages]}
        if self.seed is not None:
            body["seed"] = self.seed
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.thinking is not None:
            body["thinking"] = {"type": "enabled" if self.thinking else "disabled"}  # формат DeepSeek V4
        return body


@dataclass(frozen=True)
class ChatResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool = False
    seconds: float = 0.0  # время ответа API при первом запросе; из кэша возвращается сохранённое, а не время чтения


def chat_request(model: str, system: str, user: str, temperature: float,
                 seed: int | None = None, json_mode: bool = False, thinking: bool | None = None) -> ChatRequest:
    return ChatRequest(model, (("system", system), ("user", user)), temperature, seed, json_mode, thinking)


def read_env_file(path: Path) -> dict[str, str]:
    """Пары KEY=VALUE из .env (без поддержки кавычек и подстановок — они не нужны)."""
    lines = path.read_text().splitlines() if path.exists() else []
    pairs = (line.split("=", 1) for line in lines if "=" in line and not line.lstrip().startswith("#"))
    return {k.strip(): v.strip() for k, v in pairs}


@dataclass
class OpenAIChat:
    """Клиент Chat Completions; ключ и адрес — из окружения или .env проекта."""
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    retry: RetryPolicy = DEFAULT_RETRY
    timeout: float = 120
    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self):
        self._client = httpx.Client(timeout=self.timeout, headers={"Authorization": f"Bearer {self.api_key}"})

    @classmethod
    def from_env(cls, env_file: Path, prefix: str = "OPENAI", base_url: str | None = None) -> "OpenAIChat":
        """Ключ и адрес провайдера из переменных `<prefix>_API_KEY` и `<prefix>_BASE_URL`: любой
        OpenAI-совместимый API (OpenAI, DeepSeek) подключается без отдельного клиента."""
        env = {**read_env_file(env_file), **os.environ}
        return cls(api_key=env[f"{prefix}_API_KEY"],
                   base_url=env.get(f"{prefix}_BASE_URL", base_url or cls.base_url))

    def complete(self, request: ChatRequest) -> ChatResponse:
        start = time.perf_counter()
        response = send_with_retries(
            lambda: self._client.post(f"{self.base_url}/chat/completions", json=request.payload()), self.retry)
        seconds = time.perf_counter() - start  # с повторами после 429/5xx: столько ждёт пользователь
        body = response.json()
        usage = body.get("usage", {})
        return ChatResponse(body["choices"][0]["message"]["content"],
                            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), seconds=seconds)


@dataclass
class CachedChat:
    """Обёртка с кэшем: один JSON-файл на запрос в `cache_dir`, имя — хэш запроса."""
    client: OpenAIChat
    cache_dir: Path

    def complete(self, request: ChatRequest) -> ChatResponse:
        path = self.cache_dir / f"{request.key()}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            return ChatResponse(**{**saved["response"], "cached": True})
        response = self.client.complete(request)
        record = {"request": asdict(request), "response": {**asdict(response), "cached": False}}
        atomic_write_text(path, json.dumps(record, ensure_ascii=False, indent=1))
        return response


def parse_json_object(content: str) -> dict:
    """JSON-объект из ответа модели (в JSON-режиме ответ — ровно объект)."""
    return json.loads(content)


def json_object_or_empty(content: str) -> dict:
    """JSON-объект из ответа модели; пустой словарь, если ответ — не JSON или JSON, но не объект
    (`[]`, `null`, число): разбор ответа LLM не должен ронять прогон."""
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def json_list_field(content: str, field: str) -> list | None:
    """Список из поля JSON-объекта ответа; None, если ответ не объект, поля нет или в нём не список —
    так пустой список модели («ничего не подходит») отличается от ответа не по формату."""
    value = json_object_or_empty(content).get(field)
    return value if isinstance(value, list) else None


def is_json_object(content: str) -> bool:
    """Разбирается ли ответ модели как JSON-объект: JSON-режим этого не гарантирует."""
    try:
        return isinstance(json.loads(content), dict)
    except json.JSONDecodeError:
        return False
