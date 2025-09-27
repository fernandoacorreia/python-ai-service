from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from datetime import datetime, timezone, timedelta
from python_ai_service.main import (
    app,
    healthz,
    get_tracer,
    get_conversation_memory,
    cleanup_old_conversations,
    conversations,
    MAX_CONVERSATION_AGE_HOURS,
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


@patch("python_ai_service.main.get_langgraph_agent_with_memory")
def test_chat_success(mock_get_agent):
    """Test successful chat request with mocked LangGraph agent."""
    # Create a mock LangGraph agent that returns streaming events
    mock_agent = Mock()

    # Mock the stream method to return events with messages
    mock_message = Mock()
    mock_message.content = "Hello! How can I help you today?"
    mock_message.hasattr = Mock(return_value=True)

    mock_agent.stream.return_value = [{"messages": [mock_message]}]
    mock_get_agent.return_value = mock_agent

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
        mock_agent.stream.assert_called_once()
    finally:
        # Clean up the overrides
        app.dependency_overrides.clear()


@patch("python_ai_service.main.get_langgraph_agent_with_memory")
def test_chat_error(mock_get_agent):
    """Test chat endpoint error handling."""
    # Create a mock LangGraph agent that raises an exception
    mock_agent = Mock()
    mock_agent.stream.side_effect = Exception("API Error")
    mock_get_agent.return_value = mock_agent

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


@patch("python_ai_service.main.get_langgraph_agent_with_memory")
def test_chat_with_conversation_id(mock_get_agent):
    """Test chat request with provided conversation ID."""
    # Create a mock LangGraph agent that returns streaming events
    mock_agent = Mock()

    # Mock the stream method to return events with messages
    mock_message = Mock()
    mock_message.content = "I remember our previous conversation!"
    mock_message.hasattr = Mock(return_value=True)

    mock_agent.stream.return_value = [{"messages": [mock_message]}]
    mock_get_agent.return_value = mock_agent

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
        mock_agent.stream.assert_called_once()
    finally:
        # Clean up the overrides
        app.dependency_overrides.clear()


def test_conversation_memory_timestamp():
    """Test that conversation memory tracks creation timestamp."""
    # Clear any existing conversations
    conversations.clear()

    # Create a new conversation
    memory1 = get_conversation_memory("test-conv-1")

    # Verify it was created with a timestamp
    assert "test-conv-1" in conversations
    memory, timestamp = conversations["test-conv-1"]
    assert memory == memory1
    assert isinstance(timestamp, datetime)

    # Verify timestamp is recent (within last minute)
    now = datetime.now(timezone.utc)
    time_diff = (now - timestamp).total_seconds()
    assert time_diff < 60  # Less than 1 minute ago


def test_conversation_memory_reuse():
    """Test that existing conversation memory is reused."""
    # Clear any existing conversations
    conversations.clear()

    # Create a conversation
    memory1 = get_conversation_memory("test-conv-2")
    timestamp1 = conversations["test-conv-2"][1]

    # Get the same conversation again
    memory2 = get_conversation_memory("test-conv-2")
    timestamp2 = conversations["test-conv-2"][1]

    # Verify it's the same memory and timestamp
    assert memory1 == memory2
    assert timestamp1 == timestamp2


def test_cleanup_old_conversations():
    """Test cleanup of old conversations based on timestamp."""
    # Clear any existing conversations
    conversations.clear()

    # Create a recent conversation
    get_conversation_memory("recent-conv")

    # Create an old conversation by manually setting old timestamp
    old_memory = get_conversation_memory("old-conv")
    old_timestamp = datetime.now(timezone.utc) - timedelta(
        hours=MAX_CONVERSATION_AGE_HOURS + 1
    )
    conversations["old-conv"] = (old_memory, old_timestamp)

    # Verify both conversations exist
    assert len(conversations) == 2
    assert "recent-conv" in conversations
    assert "old-conv" in conversations

    # Run cleanup
    cleanup_old_conversations()

    # Verify only recent conversation remains
    assert len(conversations) == 1
    assert "recent-conv" in conversations
    assert "old-conv" not in conversations


def test_cleanup_no_old_conversations():
    """Test cleanup when no conversations are old enough."""
    # Clear any existing conversations
    conversations.clear()

    # Create recent conversations
    get_conversation_memory("conv-1")
    get_conversation_memory("conv-2")

    # Verify conversations exist
    assert len(conversations) == 2

    # Run cleanup
    cleanup_old_conversations()

    # Verify all conversations remain
    assert len(conversations) == 2
    assert "conv-1" in conversations
    assert "conv-2" in conversations
