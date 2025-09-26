from python_ai_service.main import greetings


def test_greetings():
    """Test the greetings function returns hello world message."""
    result = greetings()
    assert result == "Hello, World!"
