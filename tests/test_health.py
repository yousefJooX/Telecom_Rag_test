import pytest
from fastapi.testclient import TestClient


def test_health_endpoint():
    from main import app

    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "healthy"
    assert data.get("app_name") == "Telecom RAG API"
    assert data.get("version") == "1.0.0"
    assert data.get("docs_url") == "/docs"