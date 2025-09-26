from fastapi.testclient import TestClient
from python_ai_service.main import app, healthz

client = TestClient(app)


def test_healthz_function():
    """Test the healthz function directly."""
    result = healthz()
    assert result == {"status": "OK"}


def test_healthz_endpoint():
    """Test the healthz endpoint returns 200 OK."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}
