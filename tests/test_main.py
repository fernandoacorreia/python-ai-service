from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
from python_ai_service.main import app, healthz, get_llm

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


def test_chat_success():
    """Test successful chat request with mocked LLM."""
    # Create a mock LLM response
    mock_response = AsyncMock()
    mock_response.content = "Hello! How can I help you today?"

    # Create a mock LLM
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = mock_response

    # Override the dependency
    app.dependency_overrides[get_llm] = lambda: mock_llm

    try:
        response = client.post("/chat", json={"message": "Hello!"})

        assert response.status_code == 200
        assert response.json() == {"response": "Hello! How can I help you today?"}
        mock_llm.ainvoke.assert_called_once_with("Hello!")
    finally:
        # Clean up the override
        app.dependency_overrides.clear()


def test_chat_error():
    """Test chat endpoint error handling."""
    # Create a mock LLM that raises an exception
    mock_llm = AsyncMock()
    mock_llm.ainvoke.side_effect = Exception("API Error")

    # Override the dependency
    app.dependency_overrides[get_llm] = lambda: mock_llm

    try:
        response = client.post("/chat", json={"message": "Hello!"})

        assert response.status_code == 500
        assert "Error processing chat: API Error" in response.json()["detail"]
    finally:
        # Clean up the override
        app.dependency_overrides.clear()
