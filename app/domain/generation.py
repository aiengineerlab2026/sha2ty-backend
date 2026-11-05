from __future__ import annotations

from typing import Literal

from app.domain.errors import DomainError

Provider = Literal["meshy", "astra", "sol"]
PROVIDER_NAME: dict[str, str] = {"meshy": "Meshy", "astra": "GPT Astra", "sol": "GPT Sol"}


class QuotaExhausted(DomainError):
    def __init__(self, provider: Provider, limit: int) -> None:
        super().__init__(f"You have used all {limit} 3D generations for this account.")  # no AI provider names for vendors


def remaining(limit: int, used: int) -> int:
    return max(limit - used, 0)
