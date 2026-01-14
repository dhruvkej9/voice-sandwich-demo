import asyncio
import contextlib
import os
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableGenerator
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from starlette.staticfiles import StaticFiles

from assemblyai_stt import AssemblyAISTT
from cartesia_tts import CartesiaTTS
from events import (
    AgentChunkEvent,
    AgentEndEvent,
    ToolCallEvent,
    ToolResultEvent,
    VoiceAgentEvent,
    event_to_dict,
)
from utils import merge_async_iters

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

STATIC_DIR = Path(__file__).parent.parent.parent / "web" / "dist"

if not STATIC_DIR.exists():
    raise RuntimeError(
        f"Web build not found at {STATIC_DIR}. "
        "Run 'make build-web' or 'make dev-py' from the project root."
    )

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def add_to_order(item: str, quantity: int) -> str:
    """Add an item to the customer's sandwich order."""
    return f"Added {quantity} x {item} to the order."


def confirm_order(order_summary: str) -> str:
    """Confirm the final order with the customer."""
    return f"Order confirmed: {order_summary}. Sending to kitchen."


system_prompt = """
You are a helpful sandwich shop assistant. Your goal is to take the user's order.
Be concise and friendly.

Available toppings: lettuce, tomato, onion, pickles, mayo, mustard.
Available meats: turkey, ham, roast beef.
Available cheeses: swiss, cheddar, provolone.
"""

model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    temperature=float(os.getenv("GROQ_TEMPERATURE", "0.7")),
    max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "1024")),
)

agent = create_agent(
    model=model,
    tools=[add_to_order, confirm_order],
    checkpointer=MemorySaver(),
    system_prompt=system_prompt,
)


async def _stt_stream(
    audio_stream: AsyncIterator[bytes],
) -> AsyncIterator[VoiceAgentEvent]:
    stt = AssemblyAISTT(sample_rate=16000)

    async def send_audio():
        try:
            async for audio_chunk in audio_stream:
                await stt.send_audio(audio_chunk)
        finally:
            await stt.close()

    send_task = asyncio.create_task(send_audio())

    try:
        async for event in stt.receive_events():
            yield event
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            send_task.cancel()
            await send_task
        await stt.close()


async def _agent_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    thread_id = str(uuid4())

    async for event in event_stream:
        yield event

        if event.type == "stt_output":
            stream = agent.astream(
                {"messages": [HumanMessage(content=event.transcript)]},
                {"configurable": {"thread_id": thread_id}},
                stream_mode="messages",
            )

            async for message, metadata in stream:
                if isinstance(message, AIMessage):
                    yield AgentChunkEvent.create(message.text)
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            yield ToolCallEvent.create(
                                id=tool_call.get("id") or str(uuid4()),
                                name=tool_call.get("name", "unknown"),
                                args=tool_call.get("args", {}),
                            )

                if isinstance(message, ToolMessage):
                    yield ToolResultEvent.create(
                        tool_call_id=getattr(message, "tool_call_id", ""),
                        name=getattr(message, "name", "unknown"),
                        result=str(message.content) if message.content else "",
                    )

            yield AgentEndEvent.create()


async def _tts_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    tts = CartesiaTTS()

    async def process_upstream() -> AsyncIterator[VoiceAgentEvent]:
        buffer: list[str] = []
        async for event in event_stream:
            yield event
            if event.type == "agent_chunk":
                buffer.append(event.text)
            if event.type == "agent_end":
                await tts.send_text("".join(buffer))
                buffer = []

    try:
        async for event in merge_async_iters(process_upstream(), tts.receive_events()):
            yield event
    finally:
        await tts.close()


pipeline = (
    RunnableGenerator(_stt_stream)
    | RunnableGenerator(_agent_stream)
    | RunnableGenerator(_tts_stream)
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def websocket_audio_stream() -> AsyncIterator[bytes]:
        while True:
            data = await websocket.receive_bytes()
            yield data

    output_stream = pipeline.atransform(websocket_audio_stream())

    async for event in output_stream:
        await websocket.send_json(event_to_dict(event))


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
