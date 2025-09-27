from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
import os
import uvicorn
import logging
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from python_ai_service.tools import AVAILABLE_TOOLS

app = FastAPI()

# Configuration constants
MAX_CONVERSATION_AGE_HOURS = 24
CLEANUP_INTERVAL_MINUTES = 60

# In-memory storage for conversation memories with timestamps
conversations: Dict[str, Tuple[ConversationBufferMemory, datetime]] = {}


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
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str


def get_conversation_memory(conversation_id: str) -> ConversationBufferMemory:
    """Get or create conversation memory for the given conversation ID."""
    if conversation_id not in conversations:
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        timestamp = datetime.now(timezone.utc)
        conversations[conversation_id] = (memory, timestamp)
        logger.info(
            f"Created new conversation memory for ID: {conversation_id} at {timestamp.isoformat()}"
        )
    else:
        memory, timestamp = conversations[conversation_id]
        logger.debug(
            f"Retrieved existing conversation memory for ID: {conversation_id} created at {timestamp.isoformat()}"
        )
    return memory


def cleanup_old_conversations():
    """Clean up conversations older than MAX_CONVERSATION_AGE_HOURS."""
    current_time = datetime.now(timezone.utc)
    conversations_to_remove = []

    for conversation_id, (memory, timestamp) in conversations.items():
        # Calculate age in hours
        age_hours = (current_time - timestamp).total_seconds() / 3600

        if age_hours > MAX_CONVERSATION_AGE_HOURS:
            conversations_to_remove.append(conversation_id)
            logger.info(
                f"Marking conversation {conversation_id} for cleanup (age: {age_hours:.1f} hours)"
            )

    for conversation_id in conversations_to_remove:
        del conversations[conversation_id]
        logger.info(f"Cleaned up old conversation: {conversation_id}")

    logger.info(f"Active conversations: {len(conversations)}")


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
    return ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))


def get_agent_executor_with_memory(conversation_id: str) -> AgentExecutor:
    """Create an agent executor with conversation memory."""
    llm = get_llm()
    memory = get_conversation_memory(conversation_id)

    # Create a prompt template for the agent with memory
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a professional customer service representative for our e-commerce platform. 
        Your role is to assist customers with their orders in a concise and professional manner.
        
        You have access to the following tools:
        - query_orders: Query orders for a customer by their phone number
        - cancel_order: Cancel an order by its order ID
        
        Guidelines for customer service:
        - Always be polite, professional, and concise in your responses
        - Use the available tools to perform actions when customers request them
        - When a customer asks about their orders, you MUST ask for their phone number first before using the query_orders tool
        - When a customer wants to cancel an order, use the cancel_order tool with the order ID
        - Provide clear information about order statuses and any actions taken
        - If you cannot help with a request, politely explain what you can assist with
        - Keep responses brief and to the point while being helpful
        - Always collect the phone number from the customer before looking up their orders
        - Remember previous interactions in this conversation to provide better service
        
        Remember: You must use the available tools to perform actions - do not make up or guess information about orders. Always ask for the customer's phone number when they want to check their orders.""",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Create the agent
    agent = create_openai_tools_agent(llm, AVAILABLE_TOOLS, prompt)

    # Create the agent executor with memory
    agent_executor = AgentExecutor(
        agent=agent, tools=AVAILABLE_TOOLS, memory=memory, verbose=True
    )

    return agent_executor


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
    tracer_instance=Depends(get_tracer),
):
    with tracer_instance.start_as_current_span("chat") as span:
        span.set_attribute("endpoint", "chat")
        span.set_attribute("message_length", len(request.message))

        # Generate conversation_id if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        span.set_attribute("conversation_id", conversation_id)

        logger.info(
            f"Chat request received for conversation {conversation_id}: {request.message[:100]}..."
        )

        try:
            # Create agent executor with memory for this conversation
            agent_executor = get_agent_executor_with_memory(conversation_id)

            # Use the agent executor to process the request with conversation history
            response = await agent_executor.ainvoke({"input": request.message})
            response_content = response.get("output", "No response generated")

            span.set_attribute("response_length", len(response_content))
            logger.info(
                f"Chat response generated successfully for conversation {conversation_id}"
            )
            span.set_status(trace.Status(trace.StatusCode.OK))

            return ChatResponse(
                response=response_content, conversation_id=conversation_id
            )
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error_message", str(e))
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            logger.exception(
                f"Chat request failed for conversation {conversation_id}: {str(e)}"
            )
            raise HTTPException(
                status_code=500, detail=f"Error processing chat: {str(e)}"
            ) from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
