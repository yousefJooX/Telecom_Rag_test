import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


def test_query_empty_ticket():
    from main import app

    client = TestClient(app)
    response = client.post("/api/v1/query", json={"ticket": ""})

    assert response.status_code == 422


@patch("src.routers.query_router.rag_service")
def test_query_mocked_success(mock_rag,):
    from main import app

    mock_rag.answer_ticket.return_value = {
        "response": "Your internet is working fine.",
        "sources_count": 3,
        "execution_time": 1.5,
        "prompt_tokens": 50,
        "completion_tokens": 20,
        "total_tokens": 70,
    }

    client = TestClient(app)
    response = client.post("/api/v1/query", json={"ticket": "Internet not working"})

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Your internet is working fine."
    assert data["sources_count"] == 3
    assert "prompt_tokens" in data
    assert "completion_tokens" in data
    assert "total_tokens" in data