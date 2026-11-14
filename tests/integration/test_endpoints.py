"""
Integration tests for the FastAPI endpoints.
These tests require the app to be running or use the TestClient.
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with agent initialisation (data load, embeddings, LLM) mocked out."""
    from unittest.mock import patch

    from app.services.agent import agent

    # The app factory and the routes share this singleton, so stubbing its
    # startup methods skips the OpenAI calls without touching the routes.
    with patch.object(agent, "initialize", return_value=True), \
         patch.object(agent, "setup_chain", return_value=True):
        from app.core.application import create_app
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


def test_ask_empty_query(client):
    response = client.post("/ask", json={"query": "", "chat_history": []})
    assert response.status_code == 200
    assert "answer" in response.json()


def test_requirements_handles_blank_cells(client, tmp_path, monkeypatch):
    from app.core.config import settings

    csv = tmp_path / "reqs.csv"
    csv.write_text("name,budget\nAlice,500000\nBob,\n")
    monkeypatch.setattr(type(settings), "USER_REQUIREMENTS_FILE", property(lambda self: csv))

    response = client.get("/requirements")
    assert response.status_code == 200
    assert response.json()["requirements"] == [
        {"name": "Alice", "budget": 500000.0},
        {"name": "Bob", "budget": None},
    ]
