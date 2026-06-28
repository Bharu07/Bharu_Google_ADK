# ============================================================
# GOOGLE ADK — STEP 03: Multi-Turn Conversation (Sessions)
# Goal: See how ADK manages conversation history automatically
# Run:  python step_03_multi_turn.py
# ============================================================

# ── KEY DIFFERENCE FROM LANGCHAIN ─────────────────────────────────────────────
#
# LANGCHAIN (manual memory management):
#   messages = []
#   messages.append(HumanMessage("What is AI?"))
#   result = agent.invoke({"messages": messages})
#   messages.append(result["messages"][-1])  # YOU add AI's response
#   messages.append(HumanMessage("Give example"))  # YOU add next question
#   result = agent.invoke({"messages": messages})
#
# ADK (automatic memory via sessions):
#   runner.run(session_id=session.id, new_message=msg1)  # history auto-updates
#   runner.run(session_id=session.id, new_message=msg2)  # session remembers msg1 + reply1
#   # You NEVER manage the message list yourself!

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


# ── CREATE AGENT ──────────────────────────────────────────────────────────────

agent = Agent(
    name="conversation_agent",
    model=f"azure/{AZURE_DEPLOYMENT}",
    instruction="""You are a friendly tutor who explains concepts step by step.
    Remember what the user has said earlier in the conversation.
    Refer back to previous topics when relevant.
    Keep explanations concise but thorough.""",
)


# ── SETUP ─────────────────────────────────────────────────────────────────────

session_service = InMemorySessionService()
APP_NAME = "conversation_demo"
USER_ID = "student_1"
runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)


# ── MAIN ──────────────────────────────────────────────────────────────────────

async def main():
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)

    def chat(message: str) -> str:
        """Send a message and return the response."""
        user_msg = Content(role="user", parts=[Part(text=message)])
        response = runner.run(session_id=session.id, user_id=USER_ID, new_message=user_msg)
        for event in response:
            if event.is_final_response():
                return event.content.parts[0].text
        return ""

    print("=" * 55)
    print("GOOGLE ADK — STEP 03: Multi-Turn Conversation")
    print("=" * 55)

    # Turn 1
    print("\n[Turn 1]")
    print("User: What is Python?")
    reply = chat("What is Python?")
    print(f"Agent: {reply}")

    # Turn 2 — "its" refers to Python from Turn 1
    print("\n[Turn 2]")
    print("User: What are its main uses?")
    reply = chat("What are its main uses?")
    print(f"Agent: {reply}")

    # Turn 3
    print("\n[Turn 3]")
    print("User: Which one would you recommend for a beginner?")
    reply = chat("Which one would you recommend for a beginner to start with?")
    print(f"Agent: {reply}")

    # Turn 4 — New topic, but session still has full history
    print("\n[Turn 4]")
    print("User: Now teach me about JavaScript in one sentence.")
    reply = chat("Now teach me about JavaScript in one sentence.")
    print(f"Agent: {reply}")


asyncio.run(main())

# Turn 5 — References Turn 1 ("first language we discussed")
print("\n[Turn 5]")
print("User: How is it different from the first language we discussed?")
reply = chat("How is it different from the first language we discussed?")
print(f"Agent: {reply}")
# Agent knows: first language = Python, current = JavaScript


# ── INSPECT SESSION HISTORY ───────────────────────────────────────────────────

print("\n" + "=" * 55)
print("SESSION HISTORY (stored automatically)")
print("=" * 55)

current_session = session_service.get_session(
    app_name=APP_NAME, user_id=USER_ID, session_id=session.id
)
print(f"\nSession ID: {current_session.id}")
print(f"Total events: {len(current_session.events)}")


# ── MULTIPLE INDEPENDENT SESSIONS ────────────────────────────────────────────

print("\n" + "=" * 55)
print("MULTIPLE SESSIONS (isolated conversations)")
print("=" * 55)

# New session = new conversation (no memory of the above)
session_2 = session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
print(f"Session 1: {session.id} (remembers Python/JS)")
print(f"Session 2: {session_2.id} (fresh, knows nothing)")

msg = Content(role="user", parts=[Part(text="What did we discuss earlier?")])
response = runner.run(session_id=session_2.id, user_id=USER_ID, new_message=msg)
for event in response:
    if event.is_final_response():
        print(f"\n[Session 2] Agent: {event.content.parts[0].text}")
        # Will say "we haven't discussed anything" — because session_2 is empty!


# ── WHAT WE LEARNED ──────────────────────────────────────────────────────────
#
# 1. Same session_id = continuous conversation (agent remembers everything)
# 2. Different session_id = isolated conversation (no shared memory)
# 3. You NEVER manually build message lists like in LangChain
# 4. session_service.get_session() lets you inspect stored history
# 5. Each user can have multiple sessions (multiple conversations)
#
# NEXT: step_04_streaming.py — Real-time streaming responses
