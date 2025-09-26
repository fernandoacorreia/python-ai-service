# python-ai-service
Python FastAPI service for AI inference

## Development Setup

### Prerequisites
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

### Running the Application
```bash
# Run the FastAPI application
uv run python -m python_ai_service.main

# Or run with uvicorn directly
uv run uvicorn python_ai_service.main:app --reload
```

The application will be available at `http://localhost:8000`
- Health check endpoint: `http://localhost:8000/healthz`
- API documentation: `http://localhost:8000/docs`

### Code Formatting and Linting
```bash
# Format code with black
uv run --dev -m black .

# Lint code with ruff
uv run --dev -m ruff check --fix .
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
