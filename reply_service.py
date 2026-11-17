"""Логика генерации ответов клиентам через GigaChat API.

Модуль не зависит от Streamlit: его можно вызывать из интерфейса, скриптов
и будущих тестов. Здесь только сборка запроса, вызов GigaChat и разбор JSON.
"""

from __future__ import annotations

import json
import os
import re
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

PROJECT_ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"

DEFAULT_SCOPE = "GIGACHAT_API_PERS"
DEFAULT_MODEL = "GigaChat-2-Max"
DEFAULT_PROMPT_VERSION = "v2_rules"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_GUARD_RETRIES = 3
REQUEST_TIMEOUT = 60.0

SITUATIONS: tuple[str, ...] = (
    "жалоба",
    "вопрос",
    "благодарность",
    "возврат",
    "возражение",
)
TONES: tuple[str, ...] = ("дружелюбный", "деловой", "эмпатичный")

GUARD_PROMISE_ROOTS: tuple[str, ...] = (
    "оператив",
    "немедленн",
    "срочн",
    "скорее",
    "быстрее",
    "кратчайш",
    "ближайшее время",
    "завтра",
    "сегодня же",
    "гаранти",
)

GUARD_DATA_ROOTS: tuple[str, ...] = (
    "номер карт",
    "цифр карт",
    "документ",
    "паспорт",
    "номер заказ",
    "email",
    "e-mail",
)

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "situation_type": {
            "type": "string",
            "description": "Тип ситуации, определённый по сообщению клиента",
        },
        "replies": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Готовые к отправке варианты ответа",
        },
        "needs_manager": {
            "type": "boolean",
            "description": "Нужно ли передать обращение менеджеру",
        },
        "manager_reason": {
            "type": "string",
            "description": "Краткая причина передачи менеджеру",
        },
    },
    "required": ["situation_type", "replies", "needs_manager", "manager_reason"],
    "additionalProperties": True,
}

FALLBACK_INSTRUCTION = (
    "\n\nВерни только валидный JSON-объект без markdown и пояснений:\n"
    "{\n"
    '  "situation_type": "жалоба | вопрос | благодарность | возврат | возражение",\n'
    '  "replies": ["вариант 1", "вариант 2"],\n'
    '  "needs_manager": true или false,\n'
    '  "manager_reason": "причина или пустая строка"\n'
    "}"
)

SSL_MARKERS = (
    "certificate verify failed",
    "sslcertverificationerror",
    "certificate_verify_failed",
    "unable to get local issuer certificate",
    "self-signed certificate",
)

SITUATION_ALIASES = {
    "претензия": "жалоба",
    "недовольство": "жалоба",
    "консультация": "вопрос",
    "спасибо": "благодарность",
    "возврат средств": "возврат",
    "несогласие": "возражение",
}


class ReplyServiceError(Exception):
    """Базовая ошибка сервиса генерации ответов."""


class ConfigError(ReplyServiceError):
    """Ошибка конфигурации: .env, модель, версия промпта, сертификат."""


class InvalidRequestError(ReplyServiceError):
    """Некорректные входные данные формы."""


class GigaChatAPIError(ReplyServiceError):
    """Ошибка при обращении к GigaChat API."""

    def __init__(
        self,
        message: str,
        *,
        ssl_error: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.ssl_error = ssl_error
        self.status_code = status_code


class InvalidModelResponse(ReplyServiceError):
    """Модель вернула ответ, который не удалось разобрать по схеме."""


@dataclass(frozen=True)
class Settings:
    credentials: str
    scope: str
    model: str
    prompt_version: str
    temperature: float
    guard_retries: int
    ca_bundle: Path | None
    verify_ssl: bool


@dataclass(frozen=True)
class ReplyResult:
    replies: tuple[str, ...]
    situation_type: str
    needs_manager: bool
    manager_reason: str
    model: str
    prompt_version: str
    used_json_mode: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    warnings: tuple[str, ...] = ()
    guard_triggered: bool = False
    guard_attempts: int = 0


def _env_value(name: str) -> str:
    return (os.getenv(name) or "").strip().strip("'\"")


def _env_temperature() -> float:
    raw = _env_value("GIGACHAT_TEMPERATURE")
    if not raw:
        return DEFAULT_TEMPERATURE
    try:
        value = float(raw.replace(",", "."))
    except ValueError:
        return DEFAULT_TEMPERATURE
    if not 0.0 <= value <= 2.0:
        return DEFAULT_TEMPERATURE
    return value


def _env_guard_retries() -> int:
    raw = _env_value("GIGACHAT_GUARD_RETRIES")
    if not raw:
        return DEFAULT_GUARD_RETRIES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_GUARD_RETRIES
    return max(0, min(value, 10))


def load_settings() -> Settings:
    """Читает настройки из .env. Ключ используется, но нигде не выводится."""

    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

    ca_bundle_raw = _env_value("GIGACHAT_CA_BUNDLE")
    ca_bundle: Path | None = None
    if ca_bundle_raw:
        candidate = Path(ca_bundle_raw).expanduser()
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        ca_bundle = candidate

    verify_ssl = _env_value("GIGACHAT_VERIFY_SSL").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }

    return Settings(
        credentials=_env_value("GIGACHAT_CREDENTIALS"),
        scope=_env_value("GIGACHAT_SCOPE") or DEFAULT_SCOPE,
        model=_env_value("GIGACHAT_MODEL") or DEFAULT_MODEL,
        prompt_version=_env_value("PROMPT_VERSION") or DEFAULT_PROMPT_VERSION,
        temperature=_env_temperature(),
        guard_retries=_env_guard_retries(),
        ca_bundle=ca_bundle,
        verify_ssl=verify_ssl,
    )


def available_prompt_versions() -> tuple[str, ...]:
    """Возвращает имена доступных версий промпта (без расширения .md)."""

    if not PROMPTS_DIR.is_dir():
        return ()
    return tuple(sorted(path.stem for path in PROMPTS_DIR.glob("*.md") if path.is_file()))


def _load_prompt(version: str) -> str:
    available = available_prompt_versions()
    if version not in available:
        options = ", ".join(available) or "файлы в prompts/ не найдены"
        raise ConfigError(
            f"Версия промпта «{version}» не найдена. Доступные версии: {options}."
        )
    return (PROMPTS_DIR / f"{version}.md").read_text(encoding="utf-8").strip()


def build_user_message(
    *,
    message: str,
    situation: str,
    tone: str,
    company_context: str,
    variants: int,
) -> str:
    """Собирает пользовательскую часть запроса из данных формы."""

    context = company_context.strip() or "Контекст компании не предоставлен."
    return (
        f"Тип ситуации (выбрал пользователь): {situation}\n"
        f"Тон ответа: {tone}\n"
        f"Нужно вариантов ответа: {variants}\n\n"
        f"Контекст компании:\n{context}\n\n"
        f"<сообщение_клиента>\n{message.strip()}\n</сообщение_клиента>"
    )


def _exception_chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_ssl_error(exc: BaseException) -> bool:
    for item in _exception_chain(exc):
        if isinstance(item, ssl.SSLError):
            return True
        if any(marker in str(item).lower() for marker in SSL_MARKERS):
            return True
    return False


def _describe_error(exc: BaseException, credentials: str) -> str:
    status = getattr(exc, "status_code", None)
    if status is not None:
        body = getattr(exc, "content", None)
        body_text = (
            body.decode("utf-8", errors="replace")
            if isinstance(body, (bytes, bytearray))
            else ""
        )
        text = f"{type(exc).__name__} (HTTP {status}): {body_text}"
    else:
        text = f"{type(exc).__name__}: {exc}"

    if credentials:
        text = text.replace(credentials, "<скрыто>")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:900]


def _create_client(settings: Settings) -> GigaChat:
    kwargs: dict[str, Any] = {
        "credentials": settings.credentials,
        "scope": settings.scope,
        "model": settings.model,
        "verify_ssl_certs": settings.verify_ssl,
        "timeout": REQUEST_TIMEOUT,
    }
    if settings.verify_ssl and settings.ca_bundle is not None:
        if not settings.ca_bundle.is_file():
            raise ConfigError(
                "Файл сертификата не найден: "
                f"{settings.ca_bundle}. Проверьте путь в GIGACHAT_CA_BUNDLE."
            )
        kwargs["ca_bundle_file"] = str(settings.ca_bundle)

    try:
        return GigaChat(**kwargs)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(
            "Не удалось создать клиент GigaChat: "
            f"{_describe_error(exc, settings.credentials)}"
        ) from exc


def _request(
    client: GigaChat,
    settings: Settings,
    *,
    system_prompt: str,
    user_message: str,
    json_mode: bool,
    temperature: float,
) -> tuple[str, dict[str, int | None]]:
    messages = [
        Messages(role=MessagesRole.SYSTEM, content=system_prompt),
        Messages(role=MessagesRole.USER, content=user_message),
    ]
    chat_kwargs: dict[str, Any] = {}
    if json_mode:
        chat_kwargs["response_format"] = {
            "type": "json_schema",
            "schema": RESPONSE_SCHEMA,
            "strict": True,
        }

    try:
        chat = Chat(
            model=settings.model,
            messages=messages,
            temperature=temperature,
            **chat_kwargs,
        )
        response = client.chat(chat)
    except Exception as exc:  # noqa: BLE001
        raise GigaChatAPIError(
            _describe_error(exc, settings.credentials),
            ssl_error=_is_ssl_error(exc),
            status_code=getattr(exc, "status_code", None),
        ) from exc

    if not response.choices:
        raise InvalidModelResponse("Модель вернула ответ без вариантов (choices пуст).")

    content = response.choices[0].message.content or ""
    if not content.strip():
        raise InvalidModelResponse("Модель вернула пустой ответ.")

    usage = getattr(response, "usage", None)
    usage_data = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": (
            getattr(usage, "completion_tokens", None) if usage else None
        ),
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
    }
    return content, usage_data


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.IGNORECASE | re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        cleaned = cleaned[start : end + 1]

    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("ответ не является JSON-объектом")
    return payload


def _normalize_situation(value: Any, fallback: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in SITUATIONS:
        return raw
    for alias, target in SITUATION_ALIASES.items():
        if alias in raw:
            return target
    return fallback if fallback in SITUATIONS else "вопрос"


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "да", "1", "yes", "нужно", "требуется"}
    return False


def _strip_leading_quote(text: str) -> str:
    """Убирает markdown-цитирование, которым модель иногда начинает ответ."""

    cleaned = text.strip()
    while cleaned.startswith(">"):
        cleaned = cleaned[1:].lstrip()
    return cleaned


def _parse_reply(
    content: str,
    *,
    requested_variants: int,
    selected_situation: str,
) -> dict[str, Any]:
    try:
        payload = _extract_json_object(content)
    except (json.JSONDecodeError, ValueError) as exc:
        raise InvalidModelResponse(f"Ответ модели не является корректным JSON: {exc}") from exc

    replies_raw = payload.get("replies")
    if isinstance(replies_raw, str):
        replies_raw = [replies_raw]
    if not isinstance(replies_raw, list):
        raise InvalidModelResponse("В JSON отсутствует массив replies.")

    replies: list[str] = []
    for item in replies_raw:
        if not isinstance(item, str):
            continue
        cleaned = _strip_leading_quote(item)
        if cleaned:
            replies.append(cleaned)
    if not replies:
        raise InvalidModelResponse("В JSON нет ни одного непустого варианта ответа.")
    replies = replies[:requested_variants]

    needs_manager = _normalize_bool(payload.get("needs_manager"))
    reason = payload.get("manager_reason")
    reason_text = reason.strip() if isinstance(reason, str) else ""
    if needs_manager and not reason_text:
        reason_text = "Требуется решение менеджера."
    if not needs_manager:
        reason_text = ""

    return {
        "replies": tuple(replies),
        "situation_type": _normalize_situation(
            payload.get("situation_type"), selected_situation
        ),
        "needs_manager": needs_manager,
        "manager_reason": reason_text,
    }


def _can_fallback(error: BaseException) -> bool:
    if isinstance(error, InvalidModelResponse):
        return True
    if isinstance(error, GigaChatAPIError):
        return error.status_code in (400, 422)
    return False


def guard_root_in_text(root: str, text: str) -> bool:
    """Проверяет корень с учётом словоформ для многословных выражений."""

    parts = root.split()
    if len(parts) == 1:
        return root in text
    pattern = r"\s+".join(re.escape(part) + r"\w*" for part in parts)
    return re.search(pattern, text) is not None


def _find_guard_roots(replies: tuple[str, ...] | list[str]) -> list[str]:
    text = "\n".join(str(reply) for reply in replies).lower()
    return [
        root
        for root in GUARD_PROMISE_ROOTS + GUARD_DATA_ROOTS
        if guard_root_in_text(root, text)
    ]


def _guard_instruction(words: list[str]) -> str:
    listed = ", ".join(f"«{word}»" for word in words)
    return (
        f"\n\nПерепиши ответ клиенту без слов: {listed}. "
        "Верни только текст ответа в том же формате."
    )


def _merge_usage(
    total: dict[str, int | None],
    extra: dict[str, int | None],
) -> None:
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = extra.get(field)
        if isinstance(value, int):
            total[field] = (total.get(field) or 0) + value


def _generate_once(
    client: GigaChat,
    settings: Settings,
    *,
    system_prompt: str,
    user_message: str,
    variants: int,
    selected_situation: str,
    temperature: float | None = None,
) -> tuple[dict[str, Any], bool, dict[str, int | None]]:
    """Одна генерация: JSON-режим, при необходимости резервный промпт."""

    request_temperature = (
        settings.temperature if temperature is None else temperature
    )

    try:
        content, usage = _request(
            client,
            settings,
            system_prompt=system_prompt,
            user_message=user_message,
            json_mode=True,
            temperature=request_temperature,
        )
        parsed = _parse_reply(
            content,
            requested_variants=variants,
            selected_situation=selected_situation,
        )
        return parsed, True, usage
    except (GigaChatAPIError, InvalidModelResponse) as first_error:
        if not _can_fallback(first_error):
            raise

        try:
            content, usage = _request(
                client,
                settings,
                system_prompt=system_prompt,
                user_message=user_message + FALLBACK_INSTRUCTION,
                json_mode=False,
                temperature=request_temperature,
            )
            parsed = _parse_reply(
                content,
                requested_variants=variants,
                selected_situation=selected_situation,
            )
        except (GigaChatAPIError, InvalidModelResponse) as second_error:
            if isinstance(second_error, GigaChatAPIError):
                raise GigaChatAPIError(
                    "JSON-режим API не сработал "
                    f"({first_error}). Резервный запрос тоже завершился "
                    f"ошибкой: {second_error}",
                    ssl_error=second_error.ssl_error,
                    status_code=second_error.status_code,
                ) from second_error
            raise InvalidModelResponse(
                "JSON-режим API не сработал "
                f"({first_error}). Резервный запрос тоже не дал корректный "
                f"JSON: {second_error}"
            ) from second_error
        return parsed, False, usage


def generate_replies(
    *,
    message: str,
    situation: str,
    tone: str,
    company_context: str,
    variants: int = 1,
    prompt_version: str | None = None,
    guard_retries: int | None = None,
    settings: Settings | None = None,
) -> ReplyResult:
    """Собирает запрос, вызывает GigaChat и возвращает разобранный результат.

    prompt_version по умолчанию берётся из PROMPT_VERSION в .env.
    Модель всегда берётся из GIGACHAT_MODEL в .env.
    """

    settings = settings or load_settings()

    message = message.strip()
    if not message:
        raise InvalidRequestError("Введите сообщение клиента.")
    if situation not in SITUATIONS:
        options = ", ".join(SITUATIONS)
        raise InvalidRequestError(f"Неизвестный тип ситуации. Допустимо: {options}.")
    if tone not in TONES:
        options = ", ".join(TONES)
        raise InvalidRequestError(f"Неизвестный тон ответа. Допустимо: {options}.")
    if variants not in (1, 2, 3):
        raise InvalidRequestError("Количество вариантов должно быть от 1 до 3.")
    if not settings.credentials:
        raise ConfigError(
            "Ключ GIGACHAT_CREDENTIALS не задан. "
            "Скопируйте .env.example в .env и заполните ключ авторизации."
        )

    version = (prompt_version or settings.prompt_version).strip()
    system_prompt = _load_prompt(version)
    user_message = build_user_message(
        message=message,
        situation=situation,
        tone=tone,
        company_context=company_context,
        variants=variants,
    )

    max_retries = settings.guard_retries if guard_retries is None else guard_retries
    max_retries = max(0, min(int(max_retries), 10))

    with _create_client(settings) as client:
        parsed, used_json_mode, usage = _generate_once(
            client,
            settings,
            system_prompt=system_prompt,
            user_message=user_message,
            variants=variants,
            selected_situation=situation,
        )

        guard_triggered = False
        guard_attempts = 0
        warnings: tuple[str, ...] = ()
        found_words = _find_guard_roots(parsed["replies"])
        if found_words:
            warnings = tuple(found_words)

        while found_words and guard_attempts < max_retries:
            guard_triggered = True
            guard_attempts += 1
            retry_temperature = min(settings.temperature, 0.2)
            try:
                retry_parsed, retry_json_mode, retry_usage = _generate_once(
                    client,
                    settings,
                    system_prompt=system_prompt,
                    user_message=user_message + _guard_instruction(found_words),
                    variants=variants,
                    selected_situation=situation,
                    temperature=retry_temperature,
                )
            except ReplyServiceError:
                warnings = tuple(found_words)
                break

            parsed = retry_parsed
            used_json_mode = retry_json_mode
            _merge_usage(usage, retry_usage)
            found_words = _find_guard_roots(parsed["replies"])
            warnings = tuple(found_words)

    return ReplyResult(
        **parsed,
        model=settings.model,
        prompt_version=version,
        used_json_mode=used_json_mode,
        warnings=warnings,
        guard_triggered=guard_triggered,
        guard_attempts=guard_attempts,
        **usage,
    )
