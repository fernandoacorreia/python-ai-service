from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock, patch
from python_ai_service.main import (
    app,
    healthz,
    get_tracer,
)

client = TestClient(app)


def test_healthz_function():
    """Test the healthz function directly."""
    # Create a mock tracer
    mock_tracer = Mock()
    mock_span = Mock()
    mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
        return_value=mock_span
    )
    mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)

    result = healthz(tracer_instance=mock_tracer)
    assert result == {"status": "OK"}


def test_healthz_endpoint():
    """Test the healthz endpoint returns 200 OK."""
    # Create a mock tracer
    mock_tracer = Mock()
    mock_span = Mock()
    mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
        return_value=mock_span
    )
    mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)

    # Override the dependency
    app.dependency_overrides[get_tracer] = lambda: mock_tracer

    try:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "OK"}
    finally:
        # Clean up the override
        app.dependency_overrides.clear()


@patch("python_ai_service.main.get_agent_executor_with_memory")
def test_chat_success(mock_get_agent_executor):
    """Test successful chat request with mocked agent executor."""
    # Create a mock agent executor response
    mock_agent_response = {"output": "Hello! How can I help you today?"}

    # Create a mock agent executor
    mock_agent_executor = AsyncMock()
    mock_agent_executor.ainvoke.return_value = mock_agent_response
    mock_get_agent_executor.return_value = mock_agent_executor

    # Create a mock tracer
    mock_tracer = Mock()
    mock_span = Mock()
    mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
        return_value=mock_span
    )
    mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)

    # Override the dependencies
    app.dependency_overrides[get_tracer] = lambda: mock_tracer

    try:
        response = client.post("/chat", json={"message": "Hello!"})

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["response"] == "Hello! How can I help you today?"
        assert "conversation_id" in response_data
        mock_agent_executor.ainvoke.assert_called_once_with({"input": "Hello!"})
    finally:
        # Clean up the overrides
        app.dependency_overrides.clear()


@patch("python_ai_service.main.get_agent_executor_with_memory")
def test_chat_error(mock_get_agent_executor):
    """Test chat endpoint error handling."""
    # Create a mock agent executor that raises an exception
    mock_agent_executor = AsyncMock()
    mock_agent_executor.ainvoke.side_effect = Exception("API Error")
    mock_get_agent_executor.return_value = mock_agent_executor

    # Create a mock tracer
    mock_tracer = Mock()
    mock_span = Mock()
    mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
        return_value=mock_span
    )
    mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)

    # Override the dependencies
    app.dependency_overrides[get_tracer] = lambda: mock_tracer

    try:
        response = client.post("/chat", json={"message": "Hello!"})

        assert response.status_code == 500
        assert "Error processing chat: API Error" in response.json()["detail"]
    finally:
        # Clean up the overrides
        app.dependency_overrides.clear()


@patch("python_ai_service.main.get_agent_executor_with_memory")
def test_chat_with_conversation_id(mock_get_agent_executor):
    """Test chat request with provided conversation ID."""
    # Create a mock agent executor response
    mock_agent_response = {"output": "I remember our previous conversation!"}

    # Create a mock agent executor
    mock_agent_executor = AsyncMock()
    mock_agent_executor.ainvoke.return_value = mock_agent_response
    mock_get_agent_executor.return_value = mock_agent_executor

    # Create a mock tracer
    mock_tracer = Mock()
    mock_span = Mock()
    mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
        return_value=mock_span
    )
    mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)

    # Override the dependencies
    app.dependency_overrides[get_tracer] = lambda: mock_tracer

    try:
        response = client.post(
            "/chat",
            json={
                "message": "Do you remember me?",
                "conversation_id": "test-conversation-123",
            },
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["response"] == "I remember our previous conversation!"
        assert response_data["conversation_id"] == "test-conversation-123"
        mock_agent_executor.ainvoke.assert_called_once_with(
            {"input": "Do you remember me?"}
        )
    finally:
        # Clean up the overrides
        app.dependency_overrides.clear()
