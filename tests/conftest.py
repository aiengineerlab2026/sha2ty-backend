import pytest

from app.services import vision

REAL_CURRENT_MODEL = vision.current_model


@pytest.fixture(autouse=True)
def vision_model_from_settings(monkeypatch):
    """Endpoint tests never read the admin's choice from the real database: the default model applies."""

    async def default(settings, _client):
        return settings.plan_vision_model

    monkeypatch.setattr(vision, "current_model", default)
    monkeypatch.setattr(vision, "_cached", None)
