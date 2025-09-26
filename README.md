# python-ai-service
Python FastAPI service for AI inference

## Development Setup

### Prerequisites
- Python 3.14
- [uv](https://github.com/astral-sh/uv) package manager

### Setup Development Environment
```bash
# Create virtual environment
uv venv --python 3.13

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies and create virtual environment
uv sync --dev
```

### Testing
```bash
# Run unit tests
uv run pytest -v
```

### Building
```bash
# Build wheel file
uv build
```

The wheel file will be created in the `dist/` directory.
