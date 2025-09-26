from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
import os
import uvicorn
import logging
import json
from datetime import datetime, timezone
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

app = FastAPI()


# Setup JSON logging
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        return json.dumps(log_entry)


# Configure logger
logger = logging.getLogger("python_ai_service")
logger.setLevel(logging.INFO)

# Remove existing handlers to avoid duplicates
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add console handler with JSON formatter
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)


# Pydantic models
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


# Dependency injection
_otel_initialized = False


def get_tracer():
    global _otel_initialized

    if not _otel_initialized:
        trace.set_tracer_provider(TracerProvider())
        span_processor = BatchSpanProcessor(ConsoleSpanExporter())
        trace.get_tracer_provider().add_span_processor(span_processor)
        FastAPIInstrumentor.instrument_app(app)
        _otel_initialized = True

    return trace.get_tracer(__name__)


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))


@app.get("/healthz")
def healthz(tracer_instance=Depends(get_tracer)):
    with tracer_instance.start_as_current_span("healthz") as span:
        span.set_attribute("endpoint", "healthz")
        logger.info("Health check requested")
        span.set_status(trace.Status(trace.StatusCode.OK))
        return {"status": "OK"}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    llm: ChatOpenAI = Depends(get_llm),
    tracer_instance=Depends(get_tracer),
):
    with tracer_instance.start_as_current_span("chat") as span:
        span.set_attribute("endpoint", "chat")
        span.set_attribute("message_length", len(request.message))
        logger.info(f"Chat request received: {request.message[:100]}...")

        try:
            response = await llm.ainvoke(request.message)
            span.set_attribute("response_length", len(response.content))
            logger.info("Chat response generated successfully")
            span.set_status(trace.Status(trace.StatusCode.OK))
            return ChatResponse(response=response.content)
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error_message", str(e))
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            logger.exception(f"Chat request failed: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error processing chat: {str(e)}"
            ) from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
