"""Вызовы LLM через OpenAI-совместимый Chat Completions API с кэшем ответов на диске.

Каждый запрос полностью описан `ChatRequest` (модель, сообщения, температура, seed, формат) —
его хэш служит ключом кэша, поэтому повторный прогон воспроизводит ответы и ничего не стоит.
"""

import hashlib
import json
import os
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

    def key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def payload(self) -> dict:
        body = {"model": self.model, "temperature": self.temperature,
                "messages": [{"role": r, "content": c} for r, c in self.messages]}
        if self.seed is not None:
            body["seed"] = self.seed
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        return body


@dataclass(frozen=True)
class ChatResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool = False


def chat_request(model: str, system: str, user: str, temperature: float,
                 seed: int | None = None, json_mode: bool = False) -> ChatRequest:
    return ChatRequest(model, (("system", system), ("user", user)), temperature, seed, json_mode)


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
    def from_env(cls, env_file: Path) -> "OpenAIChat":
        env = {**read_env_file(env_file), **os.environ}
        return cls(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL", cls.base_url))

    def complete(self, request: ChatRequest) -> ChatResponse:
        response = send_with_retries(
            lambda: self._client.post(f"{self.base_url}/chat/completions", json=request.payload()), self.retry)
        body = response.json()
        usage = body.get("usage", {})
        return ChatResponse(body["choices"][0]["message"]["content"],
                            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))


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
