import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


def test_ingest_wrong_extension():
    from main import app

    client = TestClient(app)
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("test.exe", b"content", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Only .txt, .md, .csv" in response.json()["detail"]


def test_ingest_valid_txt():
    from main import app

    client = TestClient(app)
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("test.txt", "Hello world\nline2", "text/plain")},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["message"] == "File ingested and indexed successfully."
    assert data["chunks_indexed"] > 0
    assert data["index_path"] == "faiss_telecom_index"


def test_ingest_valid_md():
    from main import app

    client = TestClient(app)
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("readme.md", "# Header\nContent", "text/markdown")},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["chunks_indexed"] > 0