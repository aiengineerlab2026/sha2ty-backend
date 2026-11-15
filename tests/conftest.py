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


@pytest.fixture(autouse=True)
def generator_from_default(monkeypatch, request):
    """Endpoint tests never read the admin's 3D generator from the real database: Meshy applies."""
    if request.node.module.__name__.endswith("test_generator_setting"):
        return  # that test exercises the real reading

    async def default(_settings, _client):
        return "meshy"

    from app.services import generator_setting

    monkeypatch.setattr(generator_setting, "current", default)


@pytest.fixture(autouse=True)
def no_mesh_analysis(monkeypatch, request):
    """Endpoint tests never call a real vision model for the mesh analysis: none is made (test_mesh_analysis
    exercises it with its own answers)."""
    if request.node.module.__name__.endswith("test_mesh_analysis"):
        return
    from app.api import vendor

    async def none(*_args, **_kwargs):
        return None

    monkeypatch.setattr(vendor, "analysed", none)
