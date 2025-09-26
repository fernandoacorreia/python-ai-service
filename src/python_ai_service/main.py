from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
import os
import uvicorn
import logging
import json
from datetime import datetime

app = FastAPI()


# Setup JSON logging
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(datetime.UTC).isoformat(),
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
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))


@app.get("/healthz")
def healthz():
    logger.info("Health check requested")
    return {"status": "OK"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, llm: ChatOpenAI = Depends(get_llm)):
    logger.info(f"Chat request received: {request.message[:100]}...")
    try:
        response = await llm.ainvoke(request.message)
        logger.info("Chat response generated successfully")
        return ChatResponse(response=response.content)
    except Exception as e:
        logger.exception(f"Chat request failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing chat: {str(e)}"
        ) from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
