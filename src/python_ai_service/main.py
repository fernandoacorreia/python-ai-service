from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
import os
import uvicorn

app = FastAPI()

# Pydantic models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# Dependency injection
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY")
    )


@app.get("/healthz")
def healthz():
    return {"status": "OK"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, llm: ChatOpenAI = Depends(get_llm)):
    try:
        response = await llm.ainvoke(request.message)
        return ChatResponse(response=response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
