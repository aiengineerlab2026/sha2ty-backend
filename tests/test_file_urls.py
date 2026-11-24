"""File URLs name the API's real public origin: API_PUBLIC_URL when set, else the host the request reached. A server
deployed without the setting used to save http://localhost:8010/api/files/… into every model, variant and photo."""

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import app
from app.services.cdn import REQUEST_ORIGIN, Cdn, api_origin, file_prefixes


def test_the_request_host_is_the_origin_when_no_setting_names_one():
    s = Settings(api_public_url="", cdn_public_url="")
    token = REQUEST_ORIGIN.set("https://sha2ty-773972750372.europe-west1.run.app")
    try:
        assert Cdn(s).public_url("u/m/variants/a b.png") == "https://sha2ty-773972750372.europe-west1.run.app/api/files/u/m/variants/a%20b.png"
        assert file_prefixes(s)[0] == "https://sha2ty-773972750372.europe-west1.run.app/api/files/"
    finally:
        REQUEST_ORIGIN.reset(token)
    assert api_origin(s) == "http://localhost:8000"  # no request (a script): local, never the old :8010


def test_a_configured_origin_or_public_bucket_url_wins():
    token = REQUEST_ORIGIN.set("http://evil.test")
    try:
        assert Cdn(Settings(api_public_url="https://api.sha2ty.net/")).public_url("k.png") == "https://api.sha2ty.net/api/files/k.png"
        assert Cdn(Settings(api_public_url="", cdn_public_url="https://cdn.sha2ty.net")).public_url("k.png") == "https://cdn.sha2ty.net/k.png"
    finally:
        REQUEST_ORIGIN.reset(token)


def test_the_middleware_takes_the_host_and_https_but_not_a_forwarded_host():
    seen: list[str | None] = []

    @app.get("/__origin_probe")
    def probe():
        seen.append(REQUEST_ORIGIN.get())
        return {}

    try:
        client = TestClient(app, base_url="http://sha2ty-x.a.run.app")
        client.get("/__origin_probe", headers={"x-forwarded-proto": "https", "x-forwarded-host": "evil.test"})
        assert seen == ["https://sha2ty-x.a.run.app"]
    finally:
        app.router.routes = [r for r in app.router.routes if getattr(r, "path", "") != "/__origin_probe"]
    assert REQUEST_ORIGIN.get() is None  # reset after the request
