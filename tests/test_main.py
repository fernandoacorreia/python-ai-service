from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock
from python_ai_service.main import app, healthz, get_agent_executor, get_tracer

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


def test_chat_success():
    """Test successful chat request with mocked agent executor."""
    # Create a mock tracer
    mock_tracer = Mock()
    mock_span = Mock()
    mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
        return_value=mock_span
    )
    mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)

    # Create a mock agent executor response
    mock_agent_response = {"output": "Hello! How can I help you today?"}

    # Create a mock agent executor
    mock_agent_executor = AsyncMock()
    mock_agent_executor.ainvoke.return_value = mock_agent_response

    # Override the dependencies
    app.dependency_overrides[get_tracer] = lambda: mock_tracer
    app.dependency_overrides[get_agent_executor] = lambda: mock_agent_executor

    try:
        response = client.post("/chat", json={"message": "Hello!"})

        assert response.status_code == 200
        assert response.json() == {"response": "Hello! How can I help you today?"}
        mock_agent_executor.ainvoke.assert_called_once_with({"input": "Hello!"})
    finally:
        # Clean up the overrides
        app.dependency_overrides.clear()


def test_chat_error():
    """Test chat endpoint error handling."""
    # Create a mock tracer
    mock_tracer = Mock()
    mock_span = Mock()
    mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
        return_value=mock_span
    )
    mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)

    # Create a mock agent executor that raises an exception
    mock_agent_executor = AsyncMock()
    mock_agent_executor.ainvoke.side_effect = Exception("API Error")

    # Override the dependencies
    app.dependency_overrides[get_tracer] = lambda: mock_tracer
    app.dependency_overrides[get_agent_executor] = lambda: mock_agent_executor

    try:
        response = client.post("/chat", json={"message": "Hello!"})

        assert response.status_code == 500
        assert "Error processing chat: API Error" in response.json()["detail"]
    finally:
        # Clean up the overrides
        app.dependency_overrides.clear()
