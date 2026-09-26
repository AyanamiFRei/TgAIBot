"""Конфиг бота: грузится из .env через python-dotenv."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    bot_token: str
    # Какой AI-провайдер реально используется ботом: "cloudflare" | "mistral" | "nvidia".
    # Переключается через AI_PROVIDER в .env. Все говорят по одному и тому же
    # OpenAI-совместимому протоколу (/chat/completions), поэтому обслуживаются
    # одним и тем же CloudflareClient — просто с разными base_url/api_key/model
    # (см. cloudflare_client.build_ai_client).
    ai_provider: str
    # Провайдер, который обрабатывает мультимодальные ходы (фото/стикеры). Может
    # отличаться от ai_provider — например, текст гонять через дешёвый Cloudflare,
    # а картинки — через Mistral (или наоборот). По умолчанию равен ai_provider.
    # Переключается через VISION_PROVIDER в .env.
    vision_provider: str
    cloudflare_api_key: str
    cloudflare_base_url: str
    cloudflare_model: str
    mistral_api_key: str
    mistral_base_url: str
    mistral_model: str
    nvidia_api_key: str
    nvidia_base_url: str
    nvidia_model: str
    # Резервная NVIDIA-модель: когда основная упёрлась в лимит/перегруз (429/5xx),
    # клиент автоматически переключается на неё. Пусто — фолбэка нет.
    nvidia_fallback_model: str
    history_window: int
    history_chars: int
    max_tokens: int
    temperature: float
    msg_split_delay: float
    db_path: str
    sticker_pack: str
    # Proactive messaging — бот иногда пишет первым.
    proactive: bool
    proactive_min_silence_h: int
    proactive_max_silence_h: int
    proactive_min_gap_h: float
    proactive_max_per_day: int
    proactive_min_loop_min: float
    proactive_temperature: float
    proactive_max_tokens: int

    @property
    def db_full_path(self) -> str:
        p = Path(self.db_path)
        if not p.is_absolute():
            p = PROJECT_DIR / p
        return str(p)


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on", "y"}


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val else default


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val else default


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is not None:
        return _config

    load_dotenv(PROJECT_DIR / ".env")

    bot_token = (os.getenv("BOT_TOKEN") or "").strip()

    ai_provider = (os.getenv("AI_PROVIDER") or "cloudflare").strip().lower()
    if ai_provider not in ("cloudflare", "mistral", "nvidia"):
        raise RuntimeError(
            f"AI_PROVIDER={ai_provider!r} не поддерживается. "
            "Допустимые значения: 'cloudflare', 'mistral' или 'nvidia'."
        )

    # VISION_PROVIDER — отдельный переключатель для мультимодальных ходов (фото/
    # стикеры). Не задан — используется тот же провайдер, что и для текста.
    vision_provider = (os.getenv("VISION_PROVIDER") or ai_provider).strip().lower()
    if vision_provider not in ("cloudflare", "mistral", "nvidia"):
        raise RuntimeError(
            f"VISION_PROVIDER={vision_provider!r} не поддерживается. "
            "Допустимые значения: 'cloudflare', 'mistral' или 'nvidia'."
        )

    cloudflare_api_key = (os.getenv("CLOUDFLARE_API_KEY") or "").strip()
    cloudflare_url = (os.getenv("CLOUDFLARE_USER_URL") or "").strip().rstrip("/")
    mistral_api_key = (os.getenv("MISTRAL_API_KEY") or "").strip()
    mistral_url = (os.getenv("MISTRAL_BASE_URL") or "https://api.mistral.ai/v1").strip().rstrip("/")
    nvidia_api_key = (os.getenv("NVIDIA_API_KEY") or "").strip()
    nvidia_url = (
        os.getenv("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1"
    ).strip().rstrip("/")

    missing: list[str] = []
    if not bot_token:
        missing.append("BOT_TOKEN (получите у @BotFather и впишите в .env)")
    # Обязательны только переменные провайдеров, которые реально задействованы —
    # текстовым (AI_PROVIDER) и/или vision (VISION_PROVIDER), если он отличается.
    # Не заставляем заполнять ключи для неиспользуемого API.
    used_providers = {ai_provider, vision_provider}
    if "cloudflare" in used_providers:
        if not cloudflare_api_key:
            missing.append("CLOUDFLARE_API_KEY")
        if not cloudflare_url:
            missing.append("CLOUDFLARE_USER_URL")
    if "mistral" in used_providers:
        if not mistral_api_key:
            missing.append("MISTRAL_API_KEY")
    if "nvidia" in used_providers:
        if not nvidia_api_key:
            missing.append("NVIDIA_API_KEY")
    if missing:
        raise RuntimeError(
            "Не хватает обязательных переменных в .env:\n  - "
            + "\n  - ".join(missing)
            + f"\nШаблон лежит в {PROJECT_DIR/'.env.example'}."
        )

    _config = Config(
        bot_token=bot_token,
        ai_provider=ai_provider,
        vision_provider=vision_provider,
        cloudflare_api_key=cloudflare_api_key,
        cloudflare_base_url=(
            cloudflare_url[:-4] + "/v1"
            if cloudflare_url.endswith("/run")
            else cloudflare_url
        ),
        # Основная — умная быстрая русско-способная текстовая модель-собеседник.
        # Без reasoning (чтобы не уходить в петли/клоунаду), держит persona, спокойный тон.
        cloudflare_model=(
            os.getenv("CLOUDFLARE_MODEL") or "@cf/google/gemma-4-26b-a4b-it"
        ),
        mistral_api_key=mistral_api_key,
        mistral_base_url=mistral_url,
        # mistral-small-latest — дешёвая и быстрая модель, хорошо говорит по-русски,
        # умеет vision (нужно для фото/стикеров в media.py). При желании можно
        # переключить на mistral-large-latest или pixtral-*.
        mistral_model=(os.getenv("MISTRAL_MODEL") or "mistral-small-latest"),
        nvidia_api_key=nvidia_api_key,
        nvidia_base_url=nvidia_url,
        # Основная модель на NVIDIA NIM: chat/completions тот же OpenAI-совместимый
        # эндпоинт, поэтому обслуживается тем же CloudflareClient. Vision встроен
        # в модель (мультимодальная) — фото/стикеры уходят в тот же клиент.
        nvidia_model=(os.getenv("NVIDIA_MODEL") or "z-ai/glm-5.3-flash"),
        # Резервная модель для фолбэка при 429/5xx (NVIDIA_FALLBACK_MODEL в .env).
        nvidia_fallback_model=(os.getenv("NVIDIA_FALLBACK_MODEL") or "").strip(),
        # Лёгкая vision-модель: только описывает фото/стикер в 1-2 предложения,
        # описание потом подмешивается в основной диалог как текст.
        history_window=_get_int("HISTORY_WINDOW", 400),
        history_chars=_get_int("HISTORY_CHARS", 12000),
        max_tokens=_get_int("MAX_TOKENS", 512),
        temperature=_get_float("TEMPERATURE", 0.7),
        msg_split_delay=_get_float("MSG_SPLIT_DELAY", 0.8),
        db_path=os.getenv("DB_PATH") or "bot.db",
        sticker_pack=(os.getenv("STICKER_PACK") or "").strip(),
        # Proactive: по умолчанию ВЫКЛ (PROACTIVE пусто/false). Opt-in.
        proactive=_get_bool("PROACTIVE", False),
        proactive_min_silence_h=_get_int("PROACTIVE_MIN_SILENCE_H", 3),
        proactive_max_silence_h=_get_int("PROACTIVE_MAX_SILENCE_H", 8),
        proactive_min_gap_h=_get_float("PROACTIVE_MIN_GAP_H", 4.0),
        proactive_max_per_day=_get_int("PROACTIVE_MAX_PER_DAY", 2),
        proactive_min_loop_min=_get_float("PROACTIVE_MIN_LOOP_MIN", 10.0),
        proactive_temperature=_get_float("PROACTIVE_TEMPERATURE", 0.7),
        proactive_max_tokens=_get_int("PROACTIVE_MAX_TOKENS", 256),
    )
    return _config
