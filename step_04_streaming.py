# ============================================================
# GOOGLE ADK — STEP 04: Streaming Responses
# Goal: Get agent's response word-by-word in real-time
# Run:  python step_04_streaming.py
# ============================================================

# ── KEY DIFFERENCE FROM LANGCHAIN ─────────────────────────────────────────────
#
# LANGCHAIN streaming:
#   for chunk in model.stream("Why do birds fly?"):
#       print(chunk.content, end="")
#
# ADK streaming:
#   async for event in runner.run_async(...):
#       if event.content and event.content.parts:
#           print(event.content.parts[0].text, end="")
#
# ADK uses Python's asyncio for streaming.

# ── IMPORTS ──────────────────────────────────────────────────────────────────

import os
import asyncio
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

load_dotenv()
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


# ── SETUP ─────────────────────────────────────────────────────────────────────

agent = Agent(
    name="streaming_agent",
    model=f"azure/{AZURE_DEPLOYMENT}",
    instruction="You are a storyteller. Tell engaging short stories. Keep them under 100 words.",
)

session_service = InMemorySessionService()
APP_NAME = "streaming_demo"
USER_ID = "user_1"


# ── METHOD 1: SYNCHRONOUS (full response at once) ────────────────────────────

async def run_sync():
    """Wait for the complete response (like LangChain .invoke())."""
    print("=" * 55)
    print("METHOD 1: Synchronous (full response)")
    print("=" * 55)

    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

    message = Content(role="user", parts=[Part(text="Tell me a short story about a robot.")])
    response = runner.run(session_id=session.id, user_id=USER_ID, new_message=message)

    print("\nAgent: ", end="")
    for event in response:
        if event.is_final_response():
            print(event.content.parts[0].text)


# ── METHOD 2: ASYNC STREAMING (word by word) ──────────────────────────────────

async def run_streaming():
    """Stream response tokens as they're generated (like LangChain .stream())."""
    print("\n" + "=" * 55)
    print("METHOD 2: Async Streaming (real-time)")
    print("=" * 55)

    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

    message = Content(
        role="user",
        parts=[Part(text="Tell me a short story about a cat who learned to code.")]
    )

    print("\nAgent: ", end="")

    async for event in runner.run_async(
        session_id=session.id,
        user_id=USER_ID,
        new_message=message,
    ):
        # run_async() yields events AS they happen (not just at the end)
        if event.content and event.content.parts:
            text = event.content.parts[0].text
            if text:
                print(text, end="", flush=True)
                # end="" = no newline between chunks
                # flush=True = print immediately (no buffering)

    print()  # Newline after streaming finishes


# ── METHOD 3: STREAMING WITH EVENT DETAILS ────────────────────────────────────

async def run_streaming_detailed():
    """See ALL events including tool calls (for debugging)."""
    print("\n" + "=" * 55)
    print("METHOD 3: Streaming with Event Details")
    print("=" * 55)

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers.

        Args:
            a: First number.
            b: Second number.
        """
        return a * b

    tool_agent = Agent(
        name="math_streamer",
        model=f"azure/{AZURE_DEPLOYMENT}",
        instruction="Use the multiply tool for multiplication. Be concise.",
        tools=[multiply],
    )

    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    tool_runner = Runner(agent=tool_agent, app_name=APP_NAME, session_service=session_service)

    message = Content(role="user", parts=[Part(text="What is 42 * 17?")])

    print("\nEvents (shows what agent is doing internally):")
    async for event in tool_runner.run_async(
        session_id=session.id, user_id=USER_ID, new_message=message
    ):
        author = event.author
        is_final = event.is_final_response()

        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    print(f"  [{author}] TOOL CALL: {part.function_call.name}({part.function_call.args})")
                elif hasattr(part, 'function_response') and part.function_response:
                    print(f"  [{author}] TOOL RESULT: {part.function_response.response}")
                elif hasattr(part, 'text') and part.text:
                    label = "FINAL" if is_final else "partial"
                    print(f"  [{author}] {label}: {part.text[:100]}")


# ── RUN ───────────────────────────────────────────────────────────────────────

async def main():
    await run_sync()
    await run_streaming()
    await run_streaming_detailed()

if __name__ == "__main__":
    asyncio.run(main())


# ── WHAT WE LEARNED ──────────────────────────────────────────────────────────
#
# 1. runner.run() = synchronous (like LangChain .invoke())
# 2. runner.run_async() = async streaming (like LangChain .stream())
# 3. ADK uses Python asyncio: async for event in runner.run_async(...)
# 4. Events have: content, author, is_final_response()
# 5. You can see tool calls and results in the event stream
#
# NEXT: step_05_multi_agent.py — Multiple agents + orchestrator