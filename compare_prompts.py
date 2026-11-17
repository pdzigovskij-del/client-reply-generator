#!/usr/bin/env python
"""Сравнение версий промпта на тестовых обращениях из tests/cases.json.

Запуск:
    .venv\\Scripts\\python.exe compare_prompts.py
    .venv\\Scripts\\python.exe compare_prompts.py --pause 2
    .venv\\Scripts\\python.exe compare_prompts.py --retry-failed

Результаты по одному запросу на строку дописываются в reports/results.jsonl.
Итоговая таблица и тексты ответов собираются в reports/comparison.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from reply_service import (
    GUARD_DATA_ROOTS,
    GUARD_PROMISE_ROOTS,
    available_prompt_versions,
    generate_replies,
    guard_root_in_text,
    load_settings,
)

PROJECT_ROOT = Path(__file__).resolve().parent
CASES_PATH = PROJECT_ROOT / "tests" / "cases.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_PATH = REPORTS_DIR / "results.jsonl"
REPORT_PATH = REPORTS_DIR / "comparison.md"
FINAL_REPORT_PATH = REPORTS_DIR / "final_v6.md"
STABILITY_REPORT_PATH = REPORTS_DIR / "stability.md"

PROMPT_VERSIONS = (
    "v1_short",
    "v2_rules",
    "v3_examples",
    "v4_strict",
    "v5_final",
    "v6_final",
)
DEFAULT_PAUSE = 1.5

PROMISE_PHRASES = GUARD_PROMISE_ROOTS
DATA_WORDS = GUARD_DATA_ROOTS

NUMBER_RE = re.compile(r"\d[\d   ]*(?:[.,]\d+)?")
DATE_RE = re.compile(
    r"\b\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?\b"
    r"|\b\d{1,2}\s+"
    r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|"
    r"сентября|октября|ноября|декабря)\b"
    r"|\b(?:сегодня|завтра|послезавтра)\b",
    re.IGNORECASE,
)


def load_cases(path: Path) -> tuple[str, list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    company_context = data["company_context"]
    cases = data["cases"]
    if not isinstance(company_context, str) or not isinstance(cases, list):
        raise ValueError("ожидались поля company_context (строка) и cases (список)")

    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("обращение должно быть объектом")
        for field in ("id", "message", "situation_selected", "tone"):
            if field not in case:
                raise ValueError(f"в обращении нет обязательного поля {field}")
    return company_context, cases


def load_records(path: Path) -> dict[tuple[Any, str], dict[str, Any]]:
    """Читает JSONL, игнорируя записи с пометкой repeat/forced.

    Для каждой пары остаётся последняя обычная запись.
    """

    records: dict[tuple[Any, str], dict[str, Any]] = {}
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                key = (record["case_id"], record["prompt_version"])
            except (json.JSONDecodeError, KeyError, TypeError):
                print(
                    f"Предупреждение: повреждённая строка {line_number} "
                    f"в {path.name} пропущена."
                )
                continue

            repeat = record.get("repeat")
            is_repeat = bool(record.get("forced")) or (
                isinstance(repeat, int) and repeat > 0
            )
            if is_repeat:
                continue
            records[key] = record
    return records


def append_record(path: Path, record: dict[str, Any]) -> None:
    """Дописывает строку и сбрасывает её на диск сразу после запроса."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def safe_error(exc: BaseException, credentials: str) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if credentials:
        text = text.replace(credentials, "<скрыто>")
    return re.sub(r"\s+", " ", text).strip()[:900]


def extract_numbers(text: str) -> set[str]:
    numbers: set[str] = set()
    for match in NUMBER_RE.finditer(text):
        raw = match.group(0).strip()
        normalized = re.sub(r"[   ]", "", raw).replace(",", ".")
        if normalized:
            numbers.add(normalized)
    return numbers


def extract_dates(text: str) -> set[str]:
    return {match.group(0).strip() for match in DATE_RE.finditer(text)}


def number_sort_key(value: str) -> tuple[int, Any]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def find_data_roots(text: str) -> list[str]:
    """Ищет в тексте корни запрещённых слов про данные клиента."""

    lower = text.lower()
    return [root for root in DATA_WORDS if guard_root_in_text(root, lower)]


def analyze_reply_texts(
    replies: list[str] | tuple[str, ...],
    *,
    company_context: str,
    client_message: str,
    needs_manager: bool = False,
) -> list[dict[str, Any]]:
    """Ищет обещания, новые цифры/даты и опасные просьбы в ответах."""

    combined = f"{company_context}\n{client_message}"
    allowed_numbers = extract_numbers(combined)
    combined_lower = combined.lower()

    flagged: list[dict[str, Any]] = []
    for index, reply in enumerate(replies, start=1):
        reply_text = str(reply)
        reply_lower = reply_text.lower()

        promise_phrases = [
            phrase
            for phrase in PROMISE_PHRASES
            if guard_root_in_text(phrase, reply_lower)
        ]
        unsupported_numbers = sorted(
            (
                number
                for number in extract_numbers(reply_text)
                if number not in allowed_numbers
            ),
            key=number_sort_key,
        )
        unsupported_dates = sorted(
            date
            for date in extract_dates(reply_text)
            if date.lower() not in combined_lower
        )
        manager_missing = bool(needs_manager) and "менеджер" not in reply_lower
        data_words = find_data_roots(reply_text)

        if (
            promise_phrases
            or unsupported_numbers
            or unsupported_dates
            or manager_missing
            or data_words
        ):
            flagged.append(
                {
                    "index": index,
                    "promise_phrases": promise_phrases,
                    "numbers": unsupported_numbers,
                    "dates": unsupported_dates,
                    "manager_missing": manager_missing,
                    "data_words": data_words,
                }
            )
    return flagged


def quote_block(text: str) -> str:
    lines = str(text).strip().splitlines() or [""]
    return "\n".join(f"    {line}" if line else "" for line in lines)


def yes_no(value: Any) -> str:
    if value is True:
        return "да"
    if value is False:
        return "нет"
    return "—"


def match_mark(value: bool | None) -> str:
    if value is True:
        return "✓"
    if value is False:
        return "✗"
    return "—"


def record_type_match(record: dict[str, Any], case: dict[str, Any]) -> bool | None:
    expected = case.get("expected_situation")
    if expected is None:
        return None
    result = record.get("result") or {}
    return result.get("situation_type") == expected


def record_manager_match(record: dict[str, Any], case: dict[str, Any]) -> bool | None:
    expected = case.get("expected_needs_manager")
    if expected is None:
        return None
    result = record.get("result") or {}
    return bool(result.get("needs_manager")) is bool(expected)


def record_flags(
    record: dict[str, Any],
    case: dict[str, Any],
    company_context: str,
) -> list[dict[str, Any]]:
    """Пересчитывает пометки по текущим правилам, включая старые записи."""

    result = record.get("result") or {}
    replies = result.get("replies") or []
    if not isinstance(replies, list):
        return []
    return analyze_reply_texts(
        [str(reply) for reply in replies],
        company_context=company_context,
        client_message=str(case.get("message") or ""),
        needs_manager=bool(result.get("needs_manager")),
    )


def flags_short(flags: list[dict[str, Any]]) -> str:
    if not flags:
        return "—"
    parts = []
    for flag in flags:
        kinds = []
        if flag.get("promise_phrases"):
            kinds.append("обещания")
        if flag.get("numbers"):
            kinds.append("цифры")
        if flag.get("dates"):
            kinds.append("даты")
        if flag.get("manager_missing"):
            kinds.append("нет «менеджер»")
        if flag.get("data_words"):
            kinds.append("данные")
        parts.append(f"в{flag.get('index', '?')}: {', '.join(kinds)}")
    return "⚠ " + "; ".join(parts)


def flags_detail(flags: list[dict[str, Any]]) -> str:
    if not flags:
        return "нет"

    parts = []
    for flag in flags:
        bits = []
        if flag.get("promise_phrases"):
            phrases = ", ".join(f"«{item}»" for item in flag["promise_phrases"])
            bits.append(f"слова-признаки обещаний: {phrases}")
        if flag.get("numbers"):
            numbers = ", ".join(str(item) for item in flag["numbers"])
            bits.append(f"цифры, которых нет в контексте/сообщении: {numbers}")
        if flag.get("dates"):
            dates = ", ".join(f"«{item}»" for item in flag["dates"])
            bits.append(f"даты, которых нет в контексте/сообщении: {dates}")
        if flag.get("manager_missing"):
            bits.append(
                "needs_manager = true, но в тексте нет слова «менеджер»"
            )
        if flag.get("data_words"):
            words = ", ".join(f"«{item}»" for item in flag["data_words"])
            bits.append(f"слова данных: {words}")
        parts.append(f"вариант {flag.get('index', '?')} — " + "; ".join(bits))
    return "⚠ " + " | ".join(parts)


def token_text(usage: dict[str, Any]) -> str:
    def value(field: str) -> str:
        item = usage.get(field)
        return str(item) if isinstance(item, int) else "—"

    prompt = value("prompt_tokens")
    completion = value("completion_tokens")
    total = value("total_tokens")
    if prompt == completion == total == "—":
        return "API не вернул данные"
    return f"{prompt} / {completion} / {total} (prompt / completion / total)"


def version_stats(
    cases: list[dict[str, Any]],
    version: str,
    records: dict[tuple[Any, str], dict[str, Any]],
    company_context: str,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "total": len(cases),
        "ok": 0,
        "errors": 0,
        "missing": 0,
        "type_matches": 0,
        "manager_matches": 0,
        "flags": 0,
        "manager_missing": 0,
        "data_words": 0,
        "guard_triggered": 0,
        "guard_attempts": 0,
        "json_mode": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "token_fields": set(),
    }

    for case in cases:
        record = records.get((case["id"], version))
        if record is None:
            stats["missing"] += 1
            continue
        if not record.get("ok"):
            stats["errors"] += 1
            continue

        stats["ok"] += 1
        if record_type_match(record, case) is True:
            stats["type_matches"] += 1
        if record_manager_match(record, case) is True:
            stats["manager_matches"] += 1
        flags = record_flags(record, case, company_context)
        stats["flags"] += len(flags)
        for flag in flags:
            if flag.get("manager_missing"):
                stats["manager_missing"] += 1
            if flag.get("data_words"):
                stats["data_words"] += 1

        result = record.get("result") or {}
        if result.get("guard_triggered"):
            stats["guard_triggered"] += 1
        attempts = result.get("guard_attempts")
        if not isinstance(attempts, int):
            attempts = 1 if result.get("guard_triggered") else 0
        stats["guard_attempts"] += attempts
        if result.get("used_json_mode"):
            stats["json_mode"] += 1

        usage = result.get("usage") or {}
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(field)
            if isinstance(value, int):
                stats[field] += value
                stats["token_fields"].add(field)

    return stats


def token_sums_text(stats: dict[str, Any]) -> str:
    if not stats["token_fields"]:
        return "API не вернул"
    parts = []
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = stats[field]
        parts.append(str(value) if field in stats["token_fields"] else "—")
    return " / ".join(parts)


def matrix_cell(
    record: dict[str, Any] | None,
    case: dict[str, Any],
    company_context: str,
) -> str:
    if record is None:
        return "_нет данных_"
    if not record.get("ok"):
        return "ошибка"

    result = record.get("result") or {}
    detected = result.get("situation_type") or "—"
    type_mark = match_mark(record_type_match(record, case))
    manager_mark = match_mark(record_manager_match(record, case))
    flags = record_flags(record, case, company_context)

    return (
        f"тип: {detected} {type_mark}<br>"
        f"менеджер: {yes_no(bool(result.get('needs_manager')))} {manager_mark}<br>"
        f"пометки: {flags_short(flags)}"
    )


def build_report(
    *,
    cases: list[dict[str, Any]],
    versions: tuple[str, ...],
    records: dict[tuple[Any, str], dict[str, Any]],
    company_context: str,
    model: str,
) -> str:
    present = 0
    errors = 0
    for case in cases:
        for version in versions:
            record = records.get((case["id"], version))
            if record is None:
                continue
            present += 1
            if not record.get("ok"):
                errors += 1

    lines: list[str] = [
        "# Сравнение версий промпта",
        "",
        f"- **Модель:** `{model}`",
        f"- **Дата отчёта:** {datetime.now():%d.%m.%Y %H:%M:%S}",
        f"- **Обращений:** {len(cases)}",
        f"- **Версий промпта:** {', '.join(versions)}",
        f"- **Пар в `{RESULTS_PATH.name}`:** {present} из "
        f"{len(cases) * len(versions)}",
        f"- **Ошибок:** {errors}",
        "",
        "## Таблица: обращение × версия",
        "",
        "| № | " + " | ".join(versions) + " |",
        "|---|" + "|".join("---" for _ in versions) + "|",
    ]

    for case in cases:
        row = [str(case["id"])]
        for version in versions:
            row.append(
                matrix_cell(
                    records.get((case["id"], version)),
                    case,
                    company_context,
                )
            )
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Обращения", ""])

    for case in cases:
        lines.append(f"### № {case['id']} — {case.get('situation_selected', '—')}")
        lines.append("")
        lines.append("**Сообщение клиента:**")
        lines.append("")
        lines.append(quote_block(str(case.get("message") or "")))
        lines.append("")
        lines.append(
            f"**Ожидается:** тип `{case.get('expected_situation', '—')}`, "
            f"менеджер `{yes_no(case.get('expected_needs_manager'))}`"
        )
        lines.append("")

        checks = case.get("checks") or []
        if checks:
            lines.append("**Проверить вручную (checks):**")
            lines.extend(f"- {item}" for item in checks)
            lines.append("")

        must_not = case.get("must_not") or []
        if must_not:
            lines.append("**Нельзя (must_not):**")
            lines.extend(f"- {item}" for item in must_not)
            lines.append("")

        for version in versions:
            record = records.get((case["id"], version))
            lines.append(f"#### {version}")
            lines.append("")

            if record is None:
                lines.append("_Нет данных: запрос не выполнялся._")
                lines.append("")
                continue

            if not record.get("ok"):
                lines.append(f"**Ошибка:** `{record.get('error', '—')}`")
                lines.append("")
                continue

            result = record.get("result") or {}
            type_match = record_type_match(record, case)
            manager_match = record_manager_match(record, case)
            flags = record_flags(record, case, company_context)

            lines.append(
                f"- Тип: ожидался `{case.get('expected_situation', '—')}`, "
                f"получен `{result.get('situation_type', '—')}` "
                f"{match_mark(type_match)}"
            )
            lines.append(
                f"- Менеджер: ожидался "
                f"`{yes_no(case.get('expected_needs_manager'))}`, получен "
                f"`{yes_no(bool(result.get('needs_manager')))}` "
                f"{match_mark(manager_match)}"
            )
            lines.append(
                f"- Причина эскалации: {result.get('manager_reason') or '—'}"
            )
            lines.append(
                f"- JSON-режим: "
                f"{'да' if result.get('used_json_mode') else 'нет'}"
            )
            lines.append(
                f"- Токены: {token_text(result.get('usage') or {})}"
            )
            lines.append(
                "- Защита: "
                + ("срабатывала" if result.get("guard_triggered") else "не срабатывала")
            )
            warnings = result.get("warnings") or []
            if warnings:
                lines.append(
                    "- Предупреждение: " + ", ".join(str(item) for item in warnings)
                )
            lines.append(f"- Пометки по тексту: {flags_detail(flags)}")
            lines.append("")

            replies = result.get("replies") or []
            if not replies:
                lines.append("_Модель не вернула вариантов ответа._")
                lines.append("")
                continue
            for index, reply in enumerate(replies, start=1):
                lines.append(f"**Вариант {index}:**")
                lines.append("")
                lines.append(quote_block(str(reply)))
                lines.append("")

    lines.extend(
        [
            "## Сводка по версиям",
            "",
            "| Версия | Пар | Успешно | Ошибок | Нет данных | Тип ✓ | "
            "Менеджер ✓ | Пометок | Без «менеджер» | Запрос данных | "
            "Защита | Повторы | JSON-режим | Токены prompt / completion / total |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )

    totals = {
        "ok": 0,
        "errors": 0,
        "missing": 0,
        "type_matches": 0,
        "manager_matches": 0,
        "flags": 0,
        "manager_missing": 0,
        "data_words": 0,
        "guard_triggered": 0,
        "guard_attempts": 0,
        "json_mode": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "token_fields": set(),
    }

    for version in versions:
        stats = version_stats(cases, version, records, company_context)
        lines.append(
            f"| {version} | {stats['total']} | {stats['ok']} | "
            f"{stats['errors']} | {stats['missing']} | "
            f"{stats['type_matches']}/{stats['ok']} | "
            f"{stats['manager_matches']}/{stats['ok']} | "
            f"{stats['flags']} | {stats['manager_missing']} | "
            f"{stats['data_words']} | {stats['guard_triggered']} | "
            f"{stats['guard_attempts']} | "
            f"{stats['json_mode']}/{stats['ok']} | "
            f"{token_sums_text(stats)} |"
        )
        for field in (
            "ok",
            "errors",
            "missing",
            "type_matches",
            "manager_matches",
            "flags",
            "manager_missing",
            "data_words",
            "guard_triggered",
            "guard_attempts",
            "json_mode",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ):
            totals[field] += stats[field]
        totals["token_fields"].update(stats["token_fields"])

    total_pairs = len(cases) * len(versions)
    lines.append(
        f"| **Итого** | {total_pairs} | {totals['ok']} | "
        f"{totals['errors']} | {totals['missing']} | "
        f"{totals['type_matches']}/{totals['ok']} | "
        f"{totals['manager_matches']}/{totals['ok']} | "
        f"{totals['flags']} | {totals['manager_missing']} | "
        f"{totals['data_words']} | {totals['guard_triggered']} | "
        f"{totals['guard_attempts']} | "
        f"{totals['json_mode']}/{totals['ok']} | "
        f"{token_sums_text(totals)} |"
    )
    lines.append("")
    lines.append(
        "Пояснение: «Тип ✓» и «Менеджер ✓» считаются только по успешным "
        "запросам. Пометки включают корни слов-обещаний («оператив», "
        "«немедленн», «срочн», «скорее», «быстрее», «кратчайш», "
        "«ближайшее время», «завтра», «сегодня же»), цифры и даты, которых "
        "нет в контексте компании и сообщении клиента, отсутствие слова "
        "«менеджер» при needs_manager = true и корни данных: «номер карт», "
        "«цифр карт», «документ», «паспорт», «номер заказ», «email», "
        "«e-mail»."
    )
    lines.append(
        "«Защита» — сколько запросов потребовали повторной генерации "
        "из-за слов-обещаний или слов данных; «Повторы» — суммарное число "
        "таких перегенераций."
    )
    lines.append(
        "Токены — значения полей API prompt_tokens / completion_tokens / "
        "total_tokens. Если GigaChat кэширует системный промпт, он может "
        "возвращать precached_prompt_tokens отдельным полем; в этих суммах "
        "кэш не учитывается."
    )
    lines.append("")
    return "\n".join(lines)


def execute_pair(
    *,
    case: dict[str, Any],
    version: str,
    company_context: str,
    settings: Any,
    repeat: int = 0,
    forced: bool = False,
) -> dict[str, Any]:
    """Выполняет один запрос и сразу дописывает запись в JSONL."""

    case_id = case["id"]
    record_base: dict[str, Any] = {
        "case_id": case_id,
        "prompt_version": version,
        "model": settings.model,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "repeat": repeat,
        "forced": forced,
        "request": {
            "situation_selected": case.get("situation_selected"),
            "tone": case.get("tone"),
            "message": case.get("message"),
            "company_context": company_context,
        },
        "expected": {
            "situation": case.get("expected_situation"),
            "needs_manager": case.get("expected_needs_manager"),
        },
    }

    try:
        result = generate_replies(
            message=str(case["message"]),
            situation=str(case["situation_selected"]),
            tone=str(case["tone"]),
            company_context=company_context,
            variants=1,
            prompt_version=version,
            settings=settings,
        )
        text_flags = analyze_reply_texts(
            result.replies,
            company_context=company_context,
            client_message=str(case["message"]),
            needs_manager=result.needs_manager,
        )
        record = {
            **record_base,
            "ok": True,
            "result": {
                "replies": list(result.replies),
                "situation_type": result.situation_type,
                "needs_manager": result.needs_manager,
                "manager_reason": result.manager_reason,
                "used_json_mode": result.used_json_mode,
                "guard_triggered": result.guard_triggered,
                "guard_attempts": result.guard_attempts,
                "warnings": list(result.warnings),
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.total_tokens,
                },
            },
            "text_flags": text_flags,
        }
    except Exception as exc:  # noqa: BLE001
        record = {
            **record_base,
            "ok": False,
            "error": safe_error(exc, settings.credentials),
        }

    append_record(RESULTS_PATH, record)
    return record


def print_record_status(record: dict[str, Any]) -> None:
    if not record.get("ok"):
        print("ОШИБКА")
        print(f"    {record.get('error')}")
        return

    result = record.get("result") or {}
    tokens = result.get("usage", {}).get("total_tokens")
    tokens_text = str(tokens) if tokens is not None else "—"

    suffix = ""
    if result.get("guard_triggered"):
        suffix = f" [защита x{result.get('guard_attempts', 0)}]"
    warnings = result.get("warnings") or []
    if warnings:
        suffix += " [warnings: " + ", ".join(str(item) for item in warnings) + "]"
    print(f"OK (токены: {tokens_text}){suffix}")


def run_requests(
    *,
    cases: list[dict[str, Any]],
    versions: tuple[str, ...],
    company_context: str,
    settings: Any,
    records: dict[tuple[Any, str], dict[str, Any]],
    pause: float,
    retry_failed: bool,
) -> dict[tuple[Any, str], dict[str, Any]]:
    if retry_failed:
        failed_keys = [
            key for key, record in records.items() if not record.get("ok")
        ]
        for key in failed_keys:
            records.pop(key, None)

    pending: list[tuple[dict[str, Any], str]] = []
    for case in cases:
        for version in versions:
            if (case["id"], version) not in records:
                pending.append((case, version))

    total = len(pending)
    if total == 0:
        print("Все пары уже есть в reports/results.jsonl. Запросы не выполняются.")
        return records

    print(
        f"К запросу: {total} пар; уже есть: "
        f"{len(cases) * len(versions) - total}. "
        f"Модель: {settings.model}. Пауза: {pause:g} с."
    )

    for index, (case, version) in enumerate(pending, start=1):
        print(
            f"[{index}/{total}] №{case['id']} · {version} ...",
            end=" ",
            flush=True,
        )
        record = execute_pair(
            case=case,
            version=version,
            company_context=company_context,
            settings=settings,
        )
        records[(case["id"], version)] = record
        print_record_status(record)

        if pause > 0 and index < total:
            time.sleep(pause)

    return records


def run_forced_requests(
    *,
    cases: list[dict[str, Any]],
    versions: tuple[str, ...],
    company_context: str,
    settings: Any,
    repeat: int,
    pause: float,
) -> list[dict[str, Any]]:
    total = len(cases) * len(versions) * repeat
    results: list[dict[str, Any]] = []
    index = 0

    print(
        f"Принудительные повторы: {len(cases)} обращений × "
        f"{len(versions)} версий × {repeat} повторов = {total} запросов."
    )

    for repeat_index in range(1, repeat + 1):
        for case in cases:
            for version in versions:
                index += 1
                print(
                    f"[{index}/{total}] повтор {repeat_index}/{repeat} · "
                    f"№{case['id']} · {version} ...",
                    end=" ",
                    flush=True,
                )
                record = execute_pair(
                    case=case,
                    version=version,
                    company_context=company_context,
                    settings=settings,
                    repeat=repeat_index,
                    forced=True,
                )
                results.append(record)
                print_record_status(record)

                if pause > 0 and index < total:
                    time.sleep(pause)

    return results


def build_forced_report(
    *,
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    versions: tuple[str, ...],
    model: str,
) -> str:
    """Собирает отдельный отчёт по принудительному прогону."""

    ok_records = [record for record in records if record.get("ok")]
    guarded = [
        record
        for record in ok_records
        if (record.get("result") or {}).get("guard_triggered")
    ]
    attempts = sum(
        int((record.get("result") or {}).get("guard_attempts") or 0)
        for record in ok_records
    )
    warned = [
        record
        for record in ok_records
        if (record.get("result") or {}).get("warnings")
    ]

    lines: list[str] = [
        "# Финальный прогон v6_final",
        "",
        f"- **Модель:** `{model}`",
        f"- **Дата отчёта:** {datetime.now():%d.%m.%Y %H:%M:%S}",
        f"- **Обращений:** {len(cases)}",
        f"- **Версий промпта:** {', '.join(versions)}",
        f"- **Запросов:** {len(records)} (успешно {len(ok_records)})",
        f"- **Защита:** запросов с защитой {len(guarded)}, "
        f"всего перегенераций {attempts}",
        f"- **Оставшиеся warnings:** {len(warned)}",
        "",
        "Защита ищет корни слов-обещаний и данных; в предупреждениях "
        "указаны найденные корни.",
        "",
    ]

    by_case: dict[Any, list[dict[str, Any]]] = {}
    for record in records:
        by_case.setdefault(record.get("case_id"), []).append(record)

    version_order = {version: index for index, version in enumerate(versions)}

    for case in cases:
        case_records = by_case.get(case["id"], [])
        case_records.sort(
            key=lambda record: (
                version_order.get(record.get("prompt_version"), 99),
                record.get("repeat") or 0,
            )
        )

        lines.append(f"## № {case['id']} — {case.get('situation_selected', '—')}")
        lines.append("")
        lines.append("**Сообщение клиента:**")
        lines.append("")
        lines.append(quote_block(str(case.get("message") or "")))
        lines.append("")
        lines.append(
            f"**Ожидается:** тип `{case.get('expected_situation', '—')}`, "
            f"менеджер `{yes_no(case.get('expected_needs_manager'))}`"
        )
        lines.append("")

        if not case_records:
            lines.append("_Нет данных._")
            lines.append("")
            continue

        for record in case_records:
            version = record.get("prompt_version", "—")
            repeat = record.get("repeat")
            title = f"### {version}"
            if repeat:
                title += f" · повтор {repeat}"
            lines.append(title)
            lines.append("")

            if not record.get("ok"):
                lines.append(f"**Ошибка:** `{record.get('error', '—')}`")
                lines.append("")
                continue

            result = record.get("result") or {}
            type_match = record_type_match(record, case)
            manager_match = record_manager_match(record, case)

            lines.append(
                f"- Тип: ожидался `{case.get('expected_situation', '—')}`, "
                f"получен `{result.get('situation_type', '—')}` "
                f"{match_mark(type_match)}"
            )
            lines.append(
                f"- Менеджер: ожидался "
                f"`{yes_no(case.get('expected_needs_manager'))}`, получен "
                f"`{yes_no(bool(result.get('needs_manager')))}` "
                f"{match_mark(manager_match)}"
            )
            lines.append(
                f"- Причина эскалации: {result.get('manager_reason') or '—'}"
            )
            lines.append(
                "- Защита: "
                + (
                    "срабатывала"
                    if result.get("guard_triggered")
                    else "не срабатывала"
                )
                + f" (перегенераций: {result.get('guard_attempts', 0)})"
            )
            warnings = result.get("warnings") or []
            lines.append(
                "- Оставшиеся предупреждения: "
                + (
                    ", ".join(f"«{item}»" for item in warnings)
                    if warnings
                    else "нет"
                )
            )
            lines.append(
                f"- Токены: {token_text(result.get('usage') or {})}"
            )
            lines.append("")

            replies = result.get("replies") or []
            if not replies:
                lines.append("_Модель не вернула вариантов ответа._")
                lines.append("")
                continue
            for index, reply in enumerate(replies, start=1):
                lines.append(f"**Вариант {index}:**")
                lines.append("")
                lines.append(quote_block(str(reply)))
                lines.append("")

    return "\n".join(lines)


def build_stability_report(
    *,
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    versions: tuple[str, ...],
    model: str,
) -> str:
    """Все тексты повторных запусков с защитой и warnings, без выводов."""

    by_pair: dict[tuple[Any, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (record.get("case_id"), record.get("prompt_version"))
        by_pair.setdefault(key, []).append(record)

    repeats = max(
        (int(record.get("repeat") or 0) for record in records),
        default=0,
    )

    lines: list[str] = [
        "# Стабильность версий промпта",
        "",
        f"- **Модель:** `{model}`",
        f"- **Дата отчёта:** {datetime.now():%d.%m.%Y %H:%M:%S}",
        f"- **Обращений:** {len(cases)}",
        f"- **Версий промпта:** {', '.join(versions)}",
        f"- **Запусков на пару:** {repeats}",
        "",
    ]

    for case in cases:
        lines.append(f"## № {case['id']} — {case.get('situation_selected', '—')}")
        lines.append("")
        lines.append("**Сообщение клиента:**")
        lines.append("")
        lines.append(quote_block(str(case.get("message") or "")))
        lines.append("")
        lines.append(
            f"**Ожидается:** тип `{case.get('expected_situation', '—')}`, "
            f"менеджер `{yes_no(case.get('expected_needs_manager'))}`"
        )
        lines.append("")

        for version in versions:
            pair_records = by_pair.get((case["id"], version), [])
            pair_records.sort(
                key=lambda record: record.get("repeat") or 0
            )
            lines.append(f"### {version}")
            lines.append("")

            if not pair_records:
                lines.append("_Нет данных._")
                lines.append("")
                continue

            for record in pair_records:
                repeat = record.get("repeat")
                lines.append(
                    f"#### Запуск {repeat}" if repeat else "#### Запуск"
                )
                lines.append("")

                if not record.get("ok"):
                    lines.append(f"**Ошибка:** `{record.get('error', '—')}`")
                    lines.append("")
                    continue

                result = record.get("result") or {}
                lines.append(
                    f"- Тип: `{result.get('situation_type', '—')}` "
                    f"{match_mark(record_type_match(record, case))}"
                )
                lines.append(
                    f"- Менеджер: `{yes_no(bool(result.get('needs_manager')))}` "
                    f"{match_mark(record_manager_match(record, case))}"
                )
                lines.append(
                    "- Защита: "
                    + (
                        "срабатывала"
                        if result.get("guard_triggered")
                        else "не срабатывала"
                    )
                    + f" (перегенераций: {result.get('guard_attempts', 0)})"
                )
                warnings = result.get("warnings") or []
                lines.append(
                    "- Оставшиеся предупреждения: "
                    + (
                        ", ".join(f"«{item}»" for item in warnings)
                        if warnings
                        else "нет"
                    )
                )
                lines.append("")

                for index, reply in enumerate(
                    result.get("replies") or [],
                    start=1,
                ):
                    lines.append(f"**Текст {index}:**")
                    lines.append("")
                    lines.append(quote_block(str(reply)))
                    lines.append("")

        lines.append("")

    return "\n".join(lines)


def print_forced_summary(records: list[dict[str, Any]]) -> None:
    ok_records = [record for record in records if record.get("ok")]
    guarded = [
        record
        for record in ok_records
        if (record.get("result") or {}).get("guard_triggered")
    ]
    attempts = sum(
        int((record.get("result") or {}).get("guard_attempts") or 0)
        for record in ok_records
    )
    warned = [
        record
        for record in ok_records
        if (record.get("result") or {}).get("warnings")
    ]

    print("\n=== Итог повторных запусков ===")
    print(f"Запросов: {len(records)}, успешно: {len(ok_records)}")
    print(
        f"Защита срабатывала: в {len(guarded)} запросах; "
        f"всего перегенераций: {attempts}"
    )
    print(f"Warnings остались после всех попыток: {len(warned)}")

    for case_id in sorted({record["case_id"] for record in ok_records}):
        subset = [
            record for record in ok_records if record["case_id"] == case_id
        ]
        case_guarded = sum(
            1
            for record in subset
            if (record.get("result") or {}).get("guard_triggered")
        )
        case_warned = sum(
            1
            for record in subset
            if (record.get("result") or {}).get("warnings")
        )
        print(
            f"  №{case_id}: запусков {len(subset)}, "
            f"с защитой {case_guarded}, с warnings {case_warned}"
        )

    for record in ok_records:
        if record.get("case_id") != 12:
            continue
        result = record.get("result") or {}
        print(
            f"\n№12 · повтор {record.get('repeat')} · "
            f"защита={result.get('guard_triggered')} · "
            f"перегенераций={result.get('guard_attempts')} · "
            f"warnings={result.get('warnings')}"
        )
        for index, reply in enumerate(result.get("replies") or [], start=1):
            print(f"  [{index}] {reply}")


def print_summary(
    *,
    cases: list[dict[str, Any]],
    versions: tuple[str, ...],
    records: dict[tuple[Any, str], dict[str, Any]],
    company_context: str,
) -> None:
    print("\n=== Сводка по версиям ===")
    for version in versions:
        stats = version_stats(cases, version, records, company_context)
        print(
            f"{version}: успешно {stats['ok']}/{stats['total']}, "
            f"ошибок {stats['errors']}, нет данных {stats['missing']}, "
            f"тип {stats['type_matches']}/{stats['ok']}, "
            f"менеджер {stats['manager_matches']}/{stats['ok']}, "
            f"пометок {stats['flags']}, "
            f"без «менеджер» {stats['manager_missing']}, "
            f"данные {stats['data_words']}, "
            f"защита {stats['guard_triggered']} "
            f"(перегенераций {stats['guard_attempts']}), "
            f"JSON-режим {stats['json_mode']}/{stats['ok']}, "
            f"токены {token_sums_text(stats)}"
        )
    print(f"\nРезультаты: {RESULTS_PATH}")
    print(f"Отчёт: {REPORT_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Сравнить версии промпта на tests/cases.json."
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=DEFAULT_PAUSE,
        help=f"Пауза между запросами в секундах (по умолчанию {DEFAULT_PAUSE}).",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Повторить пары, которые уже записаны с ошибкой.",
    )
    parser.add_argument(
        "--versions",
        default=None,
        help=(
            "Версии через запятую, например "
            "v3_examples,v4_strict,v5_final."
        ),
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Номера обращений через запятую, например 4,11,12.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Сколько раз повторить каждый выбранный запрос.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Игнорировать уже записанные пары; результаты помечать "
            "как повторы (repeat/forced)."
        ),
    )
    args = parser.parse_args()

    if not CASES_PATH.exists():
        print(f"Ошибка: файл {CASES_PATH} не найден.")
        return 1

    try:
        company_context, cases = load_cases(CASES_PATH)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Ошибка чтения {CASES_PATH}: {exc}")
        return 1

    settings = load_settings()
    if not settings.credentials:
        print("Ошибка: GIGACHAT_CREDENTIALS не задан в .env.")
        return 1

    available = set(available_prompt_versions())
    if args.versions:
        requested = [
            version.strip()
            for version in args.versions.split(",")
            if version.strip()
        ]
        requested = list(dict.fromkeys(requested))
    else:
        requested = list(PROMPT_VERSIONS)

    versions = tuple(version for version in requested if version in available)
    missing_versions = [
        version for version in requested if version not in available
    ]
    if missing_versions:
        print(
            "Предупреждение: не найдены файлы промптов: "
            + ", ".join(missing_versions)
        )
    if not versions:
        print("Ошибка: нет доступных версий промпта.")
        return 1

    if args.repeat < 1:
        print("Ошибка: --repeat должен быть не меньше 1.")
        return 1
    if args.repeat > 1 and not args.force:
        print("Ошибка: --repeat больше 1 требует флага --force.")
        return 1

    selected_cases = cases
    if args.cases:
        try:
            requested_case_ids = [
                int(part.strip())
                for part in args.cases.split(",")
                if part.strip()
            ]
        except ValueError:
            print("Ошибка: --cases должен содержать номера через запятую.")
            return 1
        if not requested_case_ids:
            print("Ошибка: --cases не содержит номеров обращений.")
            return 1

        known_case_ids = {case["id"] for case in cases}
        unknown_case_ids = [
            case_id
            for case_id in requested_case_ids
            if case_id not in known_case_ids
        ]
        if unknown_case_ids:
            print(
                "Ошибка: неизвестные номера обращений: "
                + ", ".join(str(case_id) for case_id in unknown_case_ids)
            )
            return 1

        selected_ids = set(requested_case_ids)
        selected_cases = [
            case for case in cases if case["id"] in selected_ids
        ]

    pause = max(args.pause, 0.0)

    if args.force:
        forced_records = run_forced_requests(
            cases=selected_cases,
            versions=versions,
            company_context=company_context,
            settings=settings,
            repeat=args.repeat,
            pause=pause,
        )
        print_forced_summary(forced_records)

        if args.repeat > 1:
            stability_report = build_stability_report(
                records=forced_records,
                cases=selected_cases,
                versions=versions,
                model=settings.model,
            )
            STABILITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            STABILITY_REPORT_PATH.write_text(
                stability_report,
                encoding="utf-8",
            )
            report_path = STABILITY_REPORT_PATH
        else:
            final_report = build_forced_report(
                records=forced_records,
                cases=selected_cases,
                versions=versions,
                model=settings.model,
            )
            FINAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            FINAL_REPORT_PATH.write_text(final_report, encoding="utf-8")
            report_path = FINAL_REPORT_PATH

        print(f"\nРезультаты дописаны: {RESULTS_PATH}")
        print(f"Отчёт: {report_path}")
        print("Отчёт comparison.md в режиме --force не пересобирался.")
        return 0

    records = load_records(RESULTS_PATH)
    records = run_requests(
        cases=selected_cases,
        versions=versions,
        company_context=company_context,
        settings=settings,
        records=records,
        pause=pause,
        retry_failed=args.retry_failed,
    )

    report = build_report(
        cases=selected_cases,
        versions=versions,
        records=records,
        company_context=company_context,
        model=settings.model,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print_summary(
        cases=selected_cases,
        versions=versions,
        records=records,
        company_context=company_context,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
