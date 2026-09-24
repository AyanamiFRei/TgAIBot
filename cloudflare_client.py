"""Async OpenAI-compatible client — работает с Cloudflare Workers AI, Mistral API и NVIDIA NIM.

Все сервисы отвечают на один и тот же OpenAI-совместимый эндпоинт
`POST {base_url}/chat/completions` (включая SSE-стриминг с `data: {...}` /
`data: [DONE]`), поэтому один и тот же класс `CloudflareClient` обслуживает
любого из них — разница только в base_url / api_key / имени модели. Выбор
провайдера делается в конфиге (AI_PROVIDER=cloudflare|mistral|nvidia), а нужный
клиент собирается фабрикой `build_ai_client` ниже.

API-ключ можно указать один или несколько через запятую — при 401/403/429
клиент ротирует ключи (упёрлись в лимит одного — пробуем следующий).

Имя класса осталось историческим (CloudflareClient), чтобы не трогать сигнатуры
по всему проекту (handlers.py, proactive.py и т.д. типизированы на него) — по
сути это просто «generic OpenAI-compatible chat client».
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from config import Config

logger = logging.getLogger(__name__)


class CloudflareError(Exception):
    """AI-провайдер (Cloudflare Workers AI / Mistral) вернул ошибку."""


class CloudflareClient:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        # API-ключей может быть несколько, через запятую: при 401/403/429 клиент
        # ротирует их — если один ключ упёрся в лимит или отозван, пробуем следующий.
        self._api_keys = [k.strip() for k in api_key.split(",") if k.strip()]
        self._key_idx = 0
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(90.0, connect=15.0),
        )

    @property
    def _api_key(self) -> str:
        """Текущий активный ключ (ротируется при 401/403/429)."""
        return self._api_keys[self._key_idx % len(self._api_keys)]

    def _rotate_key(self) -> bool:
        """Переключаемся на следующий ключ. False — ключ всего один, ротировать некуда."""
        if len(self._api_keys) < 2:
            return False
        self._key_idx += 1
        return True

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        model: str | None = None,
    ) -> str:
        payload = {
            "model": model or self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        last_exc: Exception | None = None
        attempt = 0
        while attempt < 5:
            try:
                response = await self._client.post(
                    "/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning("Cloudflare transport error, attempt %d: %s", attempt + 1, exc)
                attempt += 1
                if attempt < 5:
                    await asyncio.sleep(2.0 * (2**attempt))
                    continue
                raise CloudflareError(f"не удалось связаться с моделью: {exc!s}") from exc

            if response.status_code == 200:
                return self._parse(response)

            body = response.text[:500]
            logger.error("Cloudflare %s: %s", response.status_code, body)
            last_exc = CloudflareError(f"модель ответила кодом {response.status_code}: {body}")
            attempt += 1

            # 401/403/429 — пробуем следующий ключ (если он есть) перед ретраем.
            # Ротация тоже съедает попытку: перебрали все ключи и все упёрлись —
            # выходим наружу, а не крутимся по кругу бесконечно.
            if response.status_code in (401, 403, 429) and self._rotate_key():
                logger.warning(
                    "Cloudflare %s: ключ упёрся в лимит/отказ, переключаюсь на следующий (из %d)",
                    response.status_code, len(self._api_keys),
                )
                # На другой ключ не спим retry-after — у нового ключа свой лимит.
                continue

            if response.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                await asyncio.sleep(self._retry_after(response, attempt))
                continue
            if response.status_code == 429:
                raise CloudflareError("лимит модели временно исчерпан")
            raise last_exc

        assert last_exc is not None
        raise last_exc

    async def chat_stream(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        payload = {
            "model": model or self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        last_exc: Exception | None = None
        attempt = 0
        while attempt < 5:
            try:
                async with self._client.stream(
                    "POST", "/chat/completions", json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                ) as response:
                    if response.status_code != 200:
                        body = (await response.aread()).decode("utf-8", "ignore")[:500]
                        logger.error("Cloudflare stream %s: %s", response.status_code, body)
                        last_exc = CloudflareError(
                            f"модель ответила кодом {response.status_code}: {body}"
                        )
                        attempt += 1

                        # 401/403/429 — пробуем следующий ключ (если он есть) перед ретраем.
                        # Ротация тоже съедает попытку: перебрали все ключи и все
                        # упёрлись — выходим наружу, а не крутимся по кругу.
                        if response.status_code in (401, 403, 429) and self._rotate_key():
                            logger.warning(
                                "Cloudflare stream %s: ключ упёрся в лимит/отказ, переключаюсь на следующий (из %d)",
                                response.status_code, len(self._api_keys),
                            )
                            continue

                        if response.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                            await asyncio.sleep(self._retry_after(response, attempt))
                            continue
                        if response.status_code == 429:
                            raise CloudflareError("лимит модели временно исчерпан")
                        raise last_exc

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_text = line[5:].strip()
                        if data_text == "[DONE]":
                            return
                        try:
                            data = json.loads(data_text)
                        except json.JSONDecodeError:
                            continue
                        choices = data.get("choices") or []
                        if not choices:
                            continue
                        content = (choices[0].get("delta") or {}).get("content")
                        if isinstance(content, str) and content:
                            yield content
                    return
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning(
                    "Cloudflare stream transport error, attempt %d: %s", attempt + 1, exc
                )
                attempt += 1
                if attempt < 5:
                    await asyncio.sleep(2.0 * (2**attempt))
                    continue
                raise CloudflareError(f"не удалось связаться с моделью: {exc!s}") from exc

        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _retry_after(response: httpx.Response, attempt: int) -> float:
        value = response.headers.get("retry-after")
        if value:
            try:
                return min(max(float(value), 0.5), 30.0)
            except ValueError:
                pass
        return 2.0 * (2**attempt)

    @staticmethod
    def _parse(response: httpx.Response) -> str:
        try:
            data = response.json()
        except Exception as exc:
            raise CloudflareError(f"не удалось разобрать ответ модели: {exc!s}") from exc
        choices = data.get("choices") or []
        if not choices:
            raise CloudflareError("модель вернула пустой ответ")
        content = choices[0].get("message", {}).get("content")
        return content if isinstance(content, str) else ""


def build_ai_client(cfg: "Config", provider: str | None = None) -> CloudflareClient:
    """Собирает клиент под провайдера.

    provider=None (по умолчанию) — берёт cfg.ai_provider (текстовый провайдер).
    Передайте provider явно, чтобы собрать клиент под другого провайдера —
    например, cfg.vision_provider для мультимодальных (фото/стикеры) ходов,
    если он отличается от текстового.

    Cloudflare Workers AI, Mistral API и NVIDIA NIM (integrate.api.nvidia.com) —
    OpenAI-совместимые /chat/completions эндпоинты, поэтому используется один
    и тот же класс, просто с разными base_url/api_key/model."""
    chosen = provider or cfg.ai_provider
    if chosen == "mistral":
        logger.info("AI-клиент: mistral (модель %s)", cfg.mistral_model)
        return CloudflareClient(
            base_url=cfg.mistral_base_url,
            api_key=cfg.mistral_api_key,
            model=cfg.mistral_model,
        )
    if chosen == "nvidia":
        logger.info("AI-клиент: nvidia (модель %s)", cfg.nvidia_model)
        return CloudflareClient(
            base_url=cfg.nvidia_base_url,
            api_key=cfg.nvidia_api_key,
            model=cfg.nvidia_model,
        )
    logger.info("AI-клиент: cloudflare (модель %s)", cfg.cloudflare_model)
    return CloudflareClient(
        base_url=cfg.cloudflare_base_url,
        api_key=cfg.cloudflare_api_key,
        model=cfg.cloudflare_model,
    )
