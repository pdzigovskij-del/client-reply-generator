#!/usr/bin/env python
"""Проверка доступа к GigaChat API без интерфейса.

Что делает скрипт:
1. читает ключ авторизации из .env (ключ никогда не выводится);
2. получает токен доступа;
3. задаёт модели простой вопрос на русском;
4. просит структурированный ответ (reply, needs_manager): сначала через
   встроенный JSON-режим API (response_format + json_schema), а если он
   недоступен — через явную просьбу вернуть JSON в промпте.

Запуск:
    .venv\\Scripts\\python.exe check_gigachat.py
    .venv\\Scripts\\python.exe check_gigachat.py --model GigaChat-2
    .venv\\Scripts\\python.exe check_gigachat.py --ca-bundle russian_trusted_root_ca.pem
    .venv\\Scripts\\python.exe check_gigachat.py --insecure   # только для локальной проверки
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import sys
from datetime import datetime
from typing import Any, Iterator

import httpx
from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

SIMPLE_PROMPT = "Ответь одним предложением: ты работаешь?"

STRUCTURED_PROMPT = (
    "Клиент пишет: «Здравствуйте! Когда придёт мой заказ?» "
    "Сформулируй короткий вежливый ответ клиенту и реши, нужно ли "
    "передать обращение менеджеру."
)

FALLBACK_PROMPT = (
    STRUCTURED_PROMPT
    + ' Ответь строго в формате JSON без пояснений и без markdown: '
      '{"reply": "текст ответа", "needs_manager": true или false}.'
)

JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reply": {
            "type": "string",
            "description": "Готовый ответ клиенту",
        },
        "needs_manager": {
            "type": "boolean",
            "description": "Нужно ли передать обращение менеджеру",
        },
    },
    "required": ["reply", "needs_manager"],
    "additionalProperties": True,
}

PREFERRED_MODELS = (
    "GigaChat-2-Max",
    "GigaChat-2-Pro",
    "GigaChat-2",
    "GigaChat-Pro",
    "GigaChat",
)

SSL_MARKERS = (
    "certificate verify failed",
    "sslcertverificationerror",
    "certificate_verify_failed",
    "unable to get local issuer certificate",
    "self-signed certificate",
)


def redact(text: str, secrets: list[str]) -> str:
    """Убирает секреты из любого текста перед печатью."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<скрыто>")
    return text


def exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_ssl_error(exc: BaseException) -> bool:
    for item in exception_chain(exc):
        if isinstance(item, ssl.SSLError):
            return True
        message = str(item).lower()
        if any(marker in message for marker in SSL_MARKERS):
            return True
    return False


def describe_error(exc: BaseException, secrets: list[str]) -> str:
    text = redact(f"{type(exc).__name__}: {exc}", secrets)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:900]


def print_ssl_help() -> None:
    print()
    print("Похоже, проблема в SSL-сертификате.")
    print("Причина: GigaChat использует сертификаты НУЦ Минцифры, которых нет")
    print("в стандартном наборе корневых сертификатов Python/httpx (certifi).")
    print()
    print("Правильное решение:")
    print("  1. Скачайте корневой сертификат НУЦ Минцифры:")
    print("     https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt")
    print("     (при необходимости — и выпущенный им промежуточный сертификат).")
    print("  2. Передайте его скрипту:")
    print("     .venv\\Scripts\\python.exe check_gigachat.py --ca-bundle путь\\к\\russian_trusted_root_ca.pem")
    print()
    print("Важно: Python/httpx используют собственное хранилище корней (certifi),")
    print("а не хранилище сертификатов Windows. Поэтому простой импорт сертификата")
    print("в «Доверенные корневые центры» может не помочь именно этому скрипту.")
    print()
    print("Временная мера только для локальной проверки: флаг --insecure.")
    print("Он отключает проверку сертификата и НЕ подходит для рабочего кода.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Проверка доступа к GigaChat API (ключ читается из .env)."
    )
    parser.add_argument(
        "--model",
        help="Имя модели, например GigaChat-2. По умолчанию выбирается автоматически.",
    )
    parser.add_argument(
        "--ca-bundle",
        metavar="PATH",
        help="Путь к PEM-файлу с корневым сертификатом НУЦ Минцифры.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Отключить проверку SSL-сертификата (только для локальной проверки).",
    )
    return parser.parse_args()


def pick_model(client: GigaChat, requested: str | None) -> str:
    if requested:
        return requested.strip()

    models = client.get_models()
    ids = [model.id_ for model in getattr(models, "data", []) or []]
    families = {model_id.split(":")[0].lower(): model_id.split(":")[0] for model_id in ids}

    for preferred in PREFERRED_MODELS:
        if preferred.lower() in families:
            return families[preferred.lower()]

    if ids:
        return ids[0].split(":")[0]
    return "GigaChat-2"


def extract_json_object(text: str) -> dict[str, Any]:
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


def validate_payload(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    reply = payload.get("reply")
    needs_manager = payload.get("needs_manager")

    if not isinstance(reply, str) or not reply.strip():
        return None, "поле reply отсутствует или пустое"
    if not isinstance(needs_manager, bool):
        return None, "поле needs_manager отсутствует или не является boolean"
    return {"reply": reply.strip(), "needs_manager": needs_manager}, None


def print_payload(payload: dict[str, Any]) -> None:
    print(f"   reply: {payload['reply']}")
    print(f"   needs_manager: {payload['needs_manager']}")


def request_structured_with_json_mode(
    client: GigaChat, model: str, secrets: list[str]
) -> tuple[dict[str, Any] | None, str | None]:
    payload = Chat(
        model=model,
        messages=[Messages(role=MessagesRole.USER, content=STRUCTURED_PROMPT)],
        temperature=0.2,
        response_format={
            "type": "json_schema",
            "schema": JSON_SCHEMA,
            "strict": True,
        },
    )
    try:
        response = client.chat(payload)
    except Exception as exc:  # noqa: BLE001 - нужен любой сбой API
        return None, describe_error(exc, secrets)

    content = response.choices[0].message.content or ""
    try:
        parsed = extract_json_object(content)
    except (json.JSONDecodeError, ValueError) as exc:
        preview = redact(content.strip()[:300], secrets)
        return None, f"ответ получен, но это не валидный JSON ({exc}): {preview}"

    validated, problem = validate_payload(parsed)
    if problem:
        return None, f"JSON получен, но не соответствует схеме: {problem}"
    return validated, None


def request_structured_with_prompt(
    client: GigaChat, model: str, secrets: list[str]
) -> tuple[dict[str, Any] | None, str | None]:
    payload = Chat(
        model=model,
        messages=[Messages(role=MessagesRole.USER, content=FALLBACK_PROMPT)],
        temperature=0.2,
    )
    try:
        response = client.chat(payload)
    except Exception as exc:  # noqa: BLE001
        return None, describe_error(exc, secrets)

    content = response.choices[0].message.content or ""
    try:
        parsed = extract_json_object(content)
    except (json.JSONDecodeError, ValueError) as exc:
        preview = redact(content.strip()[:300], secrets)
        return None, f"ответ не удалось разобрать как JSON ({exc}): {preview}"

    validated, problem = validate_payload(parsed)
    if problem:
        return None, f"JSON получен, но не соответствует ожиданию: {problem}"
    return validated, None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    load_dotenv()

    credentials = (os.getenv("GIGACHAT_CREDENTIALS") or "").strip().strip("'\"")
    scope = (os.getenv("GIGACHAT_SCOPE") or "").strip().strip("'\"") or "GIGACHAT_API_PERS"
    secrets = [credentials]

    print("=== Проверка доступа к GigaChat API ===")
    if not credentials:
        print("ОШИБКА: переменная GIGACHAT_CREDENTIALS не найдена в .env.")
        print("Скопируйте .env.example в .env и заполните ключ авторизации.")
        return 1

    print("Ключ авторизации: найден в .env (скрыт)")
    print(f"Scope: {scope}")

    if args.insecure:
        print("ВНИМАНИЕ: проверка SSL-сертификата ОТКЛЮЧЕНА (--insecure).")
        print("Это допустимо только для локальной проверки, не для рабочего кода.")
        print("Проверка SSL: отключена")
    elif args.ca_bundle:
        print(f"Проверка SSL: включена, дополнительный сертификат: {args.ca_bundle}")
    else:
        print("Проверка SSL: включена (стандартные корневые сертификаты)")

    client_kwargs: dict[str, Any] = {
        "credentials": credentials,
        "scope": scope,
        "verify_ssl_certs": not args.insecure,
        "timeout": 60,
    }
    if args.ca_bundle:
        client_kwargs["ca_bundle_file"] = args.ca_bundle

    try:
        with GigaChat(**client_kwargs) as client:
            # 1. Авторизация
            print("\n[1/4] Получение токена доступа...")
            token = client.get_token()
            if token is None or not getattr(token, "access_token", None):
                print("ОШИБКА: токен не получен.")
                return 1

            expires = ""
            exp = getattr(token, "expires_at", None)
            if exp:
                try:
                    moment = datetime.fromtimestamp(exp / 1000)
                    expires = f", действует до {moment:%d.%m.%Y %H:%M:%S}"
                except (OverflowError, OSError, ValueError):
                    expires = ""
            print(f"OK: токен получен (значение скрыто{expires}).")

            # 2. Выбор модели
            print("\n[2/4] Выбор модели...")
            try:
                model = pick_model(client, args.model)
            except Exception as exc:  # noqa: BLE001
                if is_ssl_error(exc):
                    print(f"ОШИБКА SSL при запросе списка моделей: {describe_error(exc, secrets)}")
                    print_ssl_help()
                    return 2
                print(f"Не удалось получить список моделей: {describe_error(exc, secrets)}")
                model = args.model or "GigaChat-2"
                print(f"Использую модель по умолчанию: {model}")
            else:
                print(f"Модель: {model}")

            # 3. Простой запрос
            print("\n[3/4] Простой запрос к модели...")
            print(f"Промпт: {SIMPLE_PROMPT}")
            try:
                simple_chat = Chat(
                    model=model,
                    messages=[Messages(role=MessagesRole.USER, content=SIMPLE_PROMPT)],
                    temperature=0.2,
                )
                simple_response = client.chat(simple_chat)
            except Exception as exc:  # noqa: BLE001
                if is_ssl_error(exc):
                    print(f"ОШИБКА SSL: {describe_error(exc, secrets)}")
                    print_ssl_help()
                    return 2
                print(f"ОШИБКА запроса: {describe_error(exc, secrets)}")
                return 1

            answer = (simple_response.choices[0].message.content or "").strip()
            model_version = getattr(simple_response, "model", None)
            print(f"OK: модель ответила{(f' (версия {model_version})' if model_version else '')}.")
            print(f"Ответ: {redact(answer, secrets)}")

            # 4. Структурированный запрос: сначала JSON-режим API
            print("\n[4/4] Структурированный ответ (reply + needs_manager)...")
            print("Попытка 1: встроенный JSON-режим API (response_format=json_schema)...")
            structured, json_mode_error = request_structured_with_json_mode(
                client, model, secrets
            )

            if structured is not None:
                print("OK: JSON-режим API сработал, ответ соответствует схеме.")
                print_payload(structured)
                print("\nИтог: доступ есть, модель отвечает, JSON-режим API работает.")
                return 0

            print(f"JSON-режим API не сработал: {json_mode_error}")
            print("Попытка 2: просьба вернуть JSON в промпте (без response_format)...")
            structured, prompt_error = request_structured_with_prompt(client, model, secrets)

            if structured is not None:
                print("OK: JSON получен через просьбу в промпте.")
                print_payload(structured)
                print(
                    "\nИтог: доступ есть, модель отвечает, но встроенный JSON-режим "
                    "не сработал — использована просьба в промпте."
                )
                return 0

            print(f"ОШИБКА: JSON не получен и через промпт: {prompt_error}")
            print("\nИтог: доступ есть, модель отвечает, структурированный ответ не получен.")
            return 1

    except Exception as exc:  # noqa: BLE001 - верхний уровень: понятный отчёт
        if is_ssl_error(exc):
            print(f"\nОШИБКА SSL: {describe_error(exc, secrets)}")
            print_ssl_help()
            return 2
        print(f"\nОШИБКА: {describe_error(exc, secrets)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
