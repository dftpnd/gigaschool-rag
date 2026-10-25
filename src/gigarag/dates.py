"""Даты в формате ISO."""

from datetime import UTC, datetime


def unix_to_iso_date(timestamp: int) -> str:
    """Unix-время → дата `YYYY-MM-DD` в UTC."""
    return datetime.fromtimestamp(timestamp, tz=UTC).date().isoformat()


def iso_date(datetime_str: str) -> str:
    """`2026-09-15T21:49:07+03:00` → `2026-09-15` (дата как в источнике, без перевода в UTC)."""
    return datetime_str[:10]
