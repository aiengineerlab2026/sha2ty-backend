from __future__ import annotations

from typing import Literal

from app.domain.errors import DomainError

Provider = Literal["meshy", "astra"]
PROVIDER_NAME: dict[str, str] = {"meshy": "Meshy", "astra": "GPT Astra"}


class QuotaExhausted(DomainError):
    def __init__(self, provider: Provider, limit: int) -> None:
        super().__init__(f"You have used all {limit} {PROVIDER_NAME[provider]} generations for this account.")


def remaining(limit: int, used: int) -> int:
    return max(limit - used, 0)
