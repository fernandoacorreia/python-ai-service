from python_ai_service.main import main


def test_main():
    """Test the main function returns hello world message."""
    result = main()
    assert result == "Hello, World!"