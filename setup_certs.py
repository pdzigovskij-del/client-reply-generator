#!/usr/bin/env python
"""Скачивает сертификаты НУЦ Минцифры и собирает CA-бандл для GigaChat.

Скрипт не читает .env и не работает с ключами API.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CERT_DIR = PROJECT_ROOT / ".certs"
OUTPUT_PATH = CERT_DIR / "gigachat_ca_bundle.pem"

ROOT_URL = "https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt"
SUB_URL = "https://gu-st.ru/content/lending/russian_trusted_sub_ca_pem.crt"

CERT_RE = re.compile(
    r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----",
    re.DOTALL,
)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


def download(url: str, timeout: float = 60.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    return data.decode("utf-8-sig")


def extract_certificates(text: str) -> list[str]:
    return [block.strip() for block in CERT_RE.findall(text)]


def validate_certificates(blocks: list[str]) -> list[bytes]:
    der_certificates: list[bytes] = []
    for block in blocks:
        der = ssl.PEM_cert_to_DER_cert(block + "\n")
        if not der:
            raise ValueError("пустой сертификат в ответе")
        der_certificates.append(der)
    return der_certificates


def fingerprint(der: bytes) -> str:
    return hashlib.sha256(der).hexdigest()


def build_bundle(force: bool) -> int:
    if OUTPUT_PATH.exists() and not force:
        print(f"Файл уже существует: {OUTPUT_PATH}")
        print("Ничего не перезаписываю. Для обновления запустите с --force.")
        return 0

    if OUTPUT_PATH.exists():
        print(f"Файл существует, перезаписываю по --force: {OUTPUT_PATH}")

    root_text = download(ROOT_URL)
    sub_text = download(SUB_URL)

    root_blocks = extract_certificates(root_text)
    sub_blocks = extract_certificates(sub_text)
    if not root_blocks:
        raise ValueError("в ответе gu-st.ru нет корневого сертификата")
    if not sub_blocks:
        raise ValueError("в ответе gu-st.ru нет промежуточного сертификата")

    root_der = validate_certificates(root_blocks)
    sub_der = validate_certificates(sub_blocks)

    bundle = "\n".join(root_blocks + sub_blocks) + "\n"
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")
    try:
        temporary_path.write_text(bundle, encoding="ascii")
        temporary_path.replace(OUTPUT_PATH)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    print("Скачано с gu-st.ru:")
    print(f"  корневой: {ROOT_URL}")
    print(f"  промежуточный: {SUB_URL}")
    print(
        "  сертификатов: "
        f"корневой {len(root_blocks)}, "
        f"промежуточный {len(sub_blocks)}"
    )
    for label, certificates in (
        ("корневой", root_der),
        ("промежуточный", sub_der),
    ):
        for index, der in enumerate(certificates, start=1):
            print(f"    {label} #{index}: SHA256 {fingerprint(der)}")

    print(f"\nСобран файл: {OUTPUT_PATH}")
    print(f"  размер: {OUTPUT_PATH.stat().st_size} байт")
    print(f"  сертификатов: {len(root_blocks) + len(sub_blocks)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Скачать сертификаты НУЦ Минцифры и собрать CA-бандл."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать существующий .certs/gigachat_ca_bundle.pem.",
    )
    args = parser.parse_args()

    try:
        return build_bundle(force=args.force)
    except (
        urllib.error.URLError,
        ssl.SSLError,
        OSError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        print(f"Ошибка: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
