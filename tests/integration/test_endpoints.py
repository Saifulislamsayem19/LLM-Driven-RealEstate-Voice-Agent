"""
Integration tests for the FastAPI endpoints.
These tests require the app to be running or use the TestClient.
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client — agent initialisation is skipped via mocking in real test suites."""
    from unittest.mock import MagicMock, patch

    mock_agent = MagicMock()
    mock_agent.df = MagicMock()
    mock_agent.df.__len__ = lambda self: 100
    mock_agent.vector_store = MagicMock()
    mock_agent.chain = MagicMock()

    with patch("app.core.agent.agent", mock_agent):
        from app.core.application import create_app
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


def test_ask_empty_query(client):
    response = client.post("/ask", json={"query": "", "chat_history": []})
    assert response.status_code == 200
    assert "answer" in response.json()
