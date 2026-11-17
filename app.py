"""Streamlit-интерфейс Client Reply Generator.

Вся работа с GigaChat API и промптами находится в reply_service.py.
Здесь только форма, вызов сервиса и отображение результата.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from reply_service import (
    SITUATIONS,
    TONES,
    GigaChatAPIError,
    ReplyServiceError,
    available_prompt_versions,
    generate_replies,
    load_settings,
)

CASES_PATH = Path(__file__).resolve().parent / "tests" / "cases.json"
EXAMPLE_WIDGET_KEY = "example_choice"

EXAMPLES_BY_LABEL: dict[str, dict] = {}
EXAMPLE_COMPANY_CONTEXT = ""


def _preview(text: str, limit: int = 50) -> str:
    """Обрезает текст обращения до limit символов для подписи в списке."""

    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


def _load_examples() -> tuple[str, list[dict]] | None:
    """Читает tests/cases.json. Возвращает None, если файла нет или он битый."""

    try:
        data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
        company_context = data["company_context"]
        raw_cases = data["cases"]
    except (OSError, ValueError, KeyError, TypeError):
        return None

    if not isinstance(company_context, str) or not isinstance(raw_cases, list):
        return None

    cases: list[dict] = []
    for item in raw_cases:
        if not isinstance(item, dict):
            continue
        case_id = item.get("id")
        message = item.get("message")
        situation = item.get("situation_selected")
        tone = item.get("tone")
        if (
            isinstance(case_id, (int, str))
            and isinstance(message, str)
            and message.strip()
            and situation in SITUATIONS
            and tone in TONES
        ):
            cases.append(
                {
                    "id": case_id,
                    "message": message,
                    "situation_selected": situation,
                    "tone": tone,
                }
            )

    if not cases:
        return None
    return company_context, cases


def _apply_example() -> None:
    """on_change: переносит выбранный пример в поля формы."""

    selected = st.session_state.get(EXAMPLE_WIDGET_KEY)
    case = EXAMPLES_BY_LABEL.get(selected)
    if case is None:
        return

    st.session_state["message_input"] = case["message"]
    st.session_state["situation_input"] = case["situation_selected"]
    st.session_state["tone_input"] = case["tone"]
    st.session_state["company_context_input"] = EXAMPLE_COMPANY_CONTEXT

st.set_page_config(
    page_title="Client Reply Generator",
    page_icon="💬",
    layout="centered",
)

settings = load_settings()
versions = available_prompt_versions()

st.session_state.setdefault("message_input", "")
st.session_state.setdefault("situation_input", SITUATIONS[0])
st.session_state.setdefault("tone_input", TONES[0])
st.session_state.setdefault("company_context_input", "")

st.title("Client Reply Generator")
st.caption("Готовые ответы клиентам на основе GigaChat")

with st.sidebar:
    st.header("Настройки")
    st.markdown(f"**Модель:** `{settings.model}`")
    st.caption("Модель задаётся в .env: GIGACHAT_MODEL")

    if versions:
        default_index = (
            versions.index(settings.prompt_version)
            if settings.prompt_version in versions
            else 0
        )
        prompt_version = st.selectbox(
            "Версия промпта",
            options=versions,
            index=default_index,
            help="По умолчанию берётся PROMPT_VERSION из .env.",
        )
    else:
        prompt_version = None
        st.error("В папке prompts/ нет файлов промптов.")

    if not settings.verify_ssl:
        st.warning(
            "Проверка SSL отключена (GIGACHAT_VERIFY_SSL=false). "
            "Допустимо только для временной локальной проверки."
        )
    if not settings.credentials:
        st.error("GIGACHAT_CREDENTIALS не задан в .env.")

examples = _load_examples()
if examples is not None:
    EXAMPLE_COMPANY_CONTEXT, example_cases = examples
    EXAMPLES_BY_LABEL.clear()

    example_labels = ["— свой текст —"]
    for case in example_cases:
        label = f"№ {case['id']} — {_preview(case['message'])}"
        example_labels.append(label)
        EXAMPLES_BY_LABEL[label] = case

    st.subheader("Загрузить пример")
    st.selectbox(
        "Обращение из tests/cases.json",
        options=example_labels,
        key=EXAMPLE_WIDGET_KEY,
        on_change=_apply_example,
    )

with st.form("reply_form"):
    message = st.text_area(
        "Сообщение клиента",
        height=160,
        placeholder="Например: «Второй день не могу войти в курс, пишет ошибку…»",
        key="message_input",
    )

    col_situation, col_tone = st.columns(2)
    with col_situation:
        situation = st.selectbox(
            "Тип ситуации",
            SITUATIONS,
            key="situation_input",
        )
    with col_tone:
        tone = st.selectbox("Тон ответа", TONES, key="tone_input")

    company_context = st.text_area(
        "Контекст компании",
        height=180,
        placeholder=(
            "Правила возврата, сроки, график работы, "
            "что нельзя обещать…"
        ),
        help=(
            "Укажите правила, сроки и преимущества, о которых можно "
            "говорить клиенту. Модель опирается только на этот текст, "
            "поэтому чем точнее контекст, тем меньше в ответах выдумок."
        ),
        key="company_context_input",
    )

    variants = st.slider("Количество вариантов", min_value=1, max_value=3, value=2)

    submitted = st.form_submit_button(
        "Сгенерировать",
        type="primary",
        disabled=not settings.credentials or not versions,
    )

if submitted:
    st.session_state.pop("result", None)
    st.session_state.pop("error", None)
    st.session_state["ssl_error"] = False

    try:
        st.session_state["result"] = generate_replies(
            message=message,
            situation=situation,
            tone=tone,
            company_context=company_context,
            variants=variants,
            prompt_version=prompt_version,
        )
    except GigaChatAPIError as exc:
        st.session_state["error"] = f"Ошибка GigaChat API: {exc}"
        st.session_state["ssl_error"] = exc.ssl_error
    except ReplyServiceError as exc:
        st.session_state["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        st.session_state["error"] = (
            f"Непредвиденная ошибка: {type(exc).__name__}: {exc}"
        )

error = st.session_state.get("error")
if error:
    st.error(error)
    if st.session_state.get("ssl_error"):
        st.info(
            "Похоже, не установлен корневой сертификат НУЦ Минцифры. "
            "Скачайте его с gu-st.ru и укажите путь в .env: "
            "GIGACHAT_CA_BUNDLE=.certs/gigachat_ca_bundle.pem. "
            "Отключать проверку сертификата (GIGACHAT_VERIFY_SSL=false) "
            "допустимо только временно для локальной проверки."
        )

result = st.session_state.get("result")
if result:
    st.divider()
    st.subheader("Результат")

    col_type, col_manager = st.columns(2)
    with col_type:
        st.markdown("**Тип ситуации**")
        st.write(result.situation_type.capitalize())
    with col_manager:
        st.markdown("**Передать менеджеру**")
        st.write("Да" if result.needs_manager else "Нет")

    if result.needs_manager:
        st.warning(f"Причина: {result.manager_reason}")

    if result.warnings:
        st.warning(
            "В ответе найдены слова-обещания или запрос данных: "
            + ", ".join(result.warnings)
            + ". Проверьте текст перед отправкой."
        )

    for index, reply in enumerate(result.replies, start=1):
        st.markdown(f"**Вариант {index}**")
        st.code(reply, language=None, wrap_lines=True)

    mode = "JSON-режим API" if result.used_json_mode else "JSON через промпт"
    st.caption(
        f"Модель: {result.model} · "
        f"Промпт: {result.prompt_version} · {mode}"
    )
