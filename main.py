import os
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from context import TWIN_SYSTEM_PROMPT
from tools import tools, handle_tool_calls


# ============================================================
# Environment
# ============================================================

load_dotenv(override=True)


# ============================================================
# Configuration
# ============================================================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Use OpenRouter's free router.
# It automatically selects an available free model.
MODEL = "openrouter/free"

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is missing from your .env file."
    )


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Taieb Digital Twin API",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-digital-twin.netlify.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request / Response Models
# ============================================================

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = Field(default_factory=list)


class ChatResponse(BaseModel):
    response: str


# ============================================================
# OpenRouter Request
# ============================================================

def call_openrouter(messages):
    """
    Send a chat request to OpenRouter.
    """

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5173",
                "X-Title": "Taieb Digital Twin",
            },
            json={
                "model": MODEL,
                "messages": messages,
                "tools": tools,
            },
            timeout=60,
        )

    except requests.RequestException as error:
        print("OpenRouter connection error:", error)

        raise HTTPException(
            status_code=502,
            detail=f"Could not connect to OpenRouter: {error}",
        )

    # --------------------------------------------------------
    # OpenRouter error
    # --------------------------------------------------------

    if not response.ok:

        print("=" * 60)
        print("OPENROUTER ERROR")
        print("Status:", response.status_code)
        print("Response:", response.text)
        print("=" * 60)

        raise HTTPException(
            status_code=502,
            detail=(
                f"OpenRouter returned "
                f"{response.status_code}: "
                f"{response.text}"
            ),
        )

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:
        return response.json()

    except ValueError:
        print("Invalid JSON returned by OpenRouter:")
        print(response.text)

        raise HTTPException(
            status_code=502,
            detail="OpenRouter returned an invalid response.",
        )

# ============================================================
# Chat Endpoint
# ============================================================

@app.post(
    "/api/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):

    # --------------------------------------------------------
    # Validate message
    # --------------------------------------------------------

    user_message = request.message.strip()

    if not user_message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    # --------------------------------------------------------
    # Conversation history
    # --------------------------------------------------------

    # Keep the latest few messages.
    #
    # Example:
    #
    # user
    # assistant
    # user
    # assistant
    # user
    # assistant
    #
    # This prevents the request from becoming unnecessarily large.

    history = [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in request.history[-1:]
    ]

    # --------------------------------------------------------
    # Messages sent to the model
    # --------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": TWIN_SYSTEM_PROMPT,
        },
        *history,
        {
            "role": "user",
            "content": user_message,
        },
    ]

    # --------------------------------------------------------
    # Agent / tool loop
    # --------------------------------------------------------

    max_tool_rounds = 5

    for round_number in range(max_tool_rounds):

        print(
            f"OpenRouter request - round {round_number + 1}"
        )

        data = call_openrouter(messages)

        # ----------------------------------------------------
        # Validate response
        # ----------------------------------------------------

        choices = data.get("choices", [])

        if not choices:
            raise HTTPException(
                status_code=502,
                detail="OpenRouter returned no choices.",
            )

        choice = choices[0]

        assistant_message = choice.get("message")

        if not assistant_message:
            raise HTTPException(
                status_code=502,
                detail="OpenRouter returned no assistant message.",
            )

        # Add assistant response to conversation
        messages.append(assistant_message)

        # ----------------------------------------------------
        # Normal assistant response
        # ----------------------------------------------------

        finish_reason = choice.get("finish_reason")

        if finish_reason != "tool_calls":

            content = assistant_message.get(
                "content",
                "",
            )

            return ChatResponse(
                response=content or "I couldn't generate a response."
            )

        # ----------------------------------------------------
        # Tool calls
        # ----------------------------------------------------

        tool_calls = assistant_message.get(
            "tool_calls",
            [],
        )

        if not tool_calls:

            return ChatResponse(
                response="I couldn't complete the tool call."
            )

        print(
            f"Tool calls detected: {len(tool_calls)}"
        )

        # Execute tools
        try:

            tool_results = handle_tool_calls(
                tool_calls
            )

        except Exception as error:

            print(
                "Tool execution error:",
                error,
            )

            raise HTTPException(
                status_code=500,
                detail=f"Tool execution failed: {error}",
            )

        # Add tool results to the conversation
        messages.extend(tool_results)

    # --------------------------------------------------------
    # Safety limit
    # --------------------------------------------------------

    return ChatResponse(
        response=(
            "I stopped because the tool-call loop "
            "reached its safety limit."
        )
    )


# ============================================================
# Health Check
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "model": MODEL,
    }