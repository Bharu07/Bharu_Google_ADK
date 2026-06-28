# ============================================================
# GOOGLE ADK — STEP 10: Dynamic Instructions
# Goal: Generate agent instructions at runtime based on context
# Run:  python step_10_dynamic_instructions.py
# ============================================================

# ── KEY CONCEPT ──────────────────────────────────────────────────────────────
#
# In Steps 01-09, we used STATIC instruction strings:
#   Agent(instruction="You are a helpful assistant.")
#
# But ADK also supports DYNAMIC instructions — a Python FUNCTION that
# generates the instruction at runtime, based on session state/context.
#
# WHY?
#   - Personalize behavior per user
#   - Change behavior based on time of day
#   - Include live data (like user's subscription status)
#   - Adjust tone/style based on context
#   - Feature flags (enable/disable capabilities)
#
# LANGCHAIN EQUIVALENT:
#   In LangChain, you'd modify the messages list manually before calling invoke().
#   ADK makes this a first-class feature via callable instructions.
#
# SYNTAX:
#   Static:  Agent(instruction="Fixed string here")
#   Dynamic: Agent(instruction=my_function)
#
#   def my_function(readonly_context) -> str:
#       # Access state, return an instruction string
#       return f"You are helping {readonly_context.state['user_name']}"

# ── IMPORTS ──────────────────────────────────────────────────────────────────

import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

load_dotenv()
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

session_service = InMemorySessionService()
APP_NAME = "dynamic_instructions_demo"
USER_ID = "user_1"


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 1: Time-Aware Agent
# Instruction changes based on time of day.
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("EXAMPLE 1: Time-Aware Dynamic Instruction")
print("=" * 60)


def time_aware_instruction(readonly_context) -> str:
    """Generate instruction based on current time of day.

    Args:
        readonly_context: ADK provides this automatically.
            - readonly_context.state → session state dict
            - Use it to read (not write) state.

    Returns:
        The instruction string for the agent to use.
    """
    hour = datetime.now().hour

    if 5 <= hour < 12:
        greeting = "Good morning"
        mood = "energetic and motivating"
        suggestion = "Suggest productive activities for the morning."
    elif 12 <= hour < 17:
        greeting = "Good afternoon"
        mood = "focused and professional"
        suggestion = "Help with work-related tasks."
    elif 17 <= hour < 21:
        greeting = "Good evening"
        mood = "relaxed and friendly"
        suggestion = "Suggest relaxing activities or help with personal projects."
    else:
        greeting = "Hello, night owl"
        mood = "calm and supportive"
        suggestion = "Be gentle, the user might be tired."

    return f"""You are a personal assistant. Your tone is {mood}.
    Start your first response with "{greeting}!"
    {suggestion}
    Current time: {datetime.now().strftime('%H:%M')}.
    Keep responses concise (2-3 sentences max)."""


# Pass the FUNCTION (not the result!) as instruction
time_agent = Agent(
    name="time_aware_agent",
    model=f"azure/{AZURE_DEPLOYMENT}",
    instruction=time_aware_instruction,
    # ^^^ This is a FUNCTION, not a string!
    # ADK calls it EVERY TIME the agent needs its instruction.
    # The instruction is freshly generated each turn.
)


async def main():

    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner = Runner(agent=time_agent, app_name=APP_NAME, session_service=session_service)

    print(f"\nCurrent time: {datetime.now().strftime('%H:%M')}")
    print("Test: 'What should I do right now?'")
    msg = Content(role="user", parts=[Part(text="What should I do right now?")])
    response = runner.run(session_id=session.id, user_id=USER_ID, new_message=msg)
    for event in response:
        if event.is_final_response():
            print(f"  Response: {event.content.parts[0].text[:200]}")


    # ══════════════════════════════════════════════════════════════════════════════
    # EXAMPLE 2: User-Profile-Aware Agent
    # Instruction changes based on session state (user profile).
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("EXAMPLE 2: User-Profile Dynamic Instruction")
    print("=" * 60)


    def profile_based_instruction(readonly_context) -> str:
        """Generate instruction based on user's profile stored in state.

        The state is pre-loaded when the session is created, or
        written by tools during the conversation.
        """
        state = readonly_context.state

        user_name = state.get("user_name", "User")
        expertise = state.get("expertise_level", "beginner")
        interests = state.get("interests", "general topics")
        language = state.get("language", "English")

        # Adjust complexity based on expertise
        if expertise == "beginner":
            complexity = "Use simple language. Avoid jargon. Give analogies."
        elif expertise == "intermediate":
            complexity = "Use some technical terms but explain complex concepts."
        else:
            complexity = "Be technical and detailed. Assume strong background knowledge."

        return f"""You are a personal tutor for {user_name}.
        Their expertise level: {expertise}.
        Their interests: {interests}.
        Respond in: {language}.

        {complexity}

        Keep responses focused and under 3 sentences unless asked to elaborate."""


    profile_agent = Agent(
        name="profile_agent",
        model=f"azure/{AZURE_DEPLOYMENT}",
        instruction=profile_based_instruction,
    )

    # ── Test with BEGINNER profile ──
    print("\n--- Beginner Profile ---")
    session_beginner = await session_service.create_session(
        app_name=APP_NAME,
        user_id="beginner_user",
        state={
            "user_name": "Bharath",
            "expertise_level": "beginner",
            "interests": "AI and machine learning",
            "language": "English",
        },
    )
    runner_b = Runner(agent=profile_agent, app_name=APP_NAME, session_service=session_service)

    msg = Content(role="user", parts=[Part(text="What is a neural network?")])
    response = runner_b.run(session_id=session_beginner.id, user_id="beginner_user", new_message=msg)
    for event in response:
        if event.is_final_response():
            print(f"  Beginner response: {event.content.parts[0].text[:200]}")

    # ── Test with EXPERT profile ──
    print("\n--- Expert Profile ---")
    session_expert = await session_service.create_session(
        app_name=APP_NAME,
        user_id="expert_user",
        state={
            "user_name": "Dr. Smith",
            "expertise_level": "expert",
            "interests": "deep learning architectures",
            "language": "English",
        },
    )
    runner_e = Runner(agent=profile_agent, app_name=APP_NAME, session_service=session_service)

    msg = Content(role="user", parts=[Part(text="What is a neural network?")])
    response = runner_e.run(session_id=session_expert.id, user_id="expert_user", new_message=msg)
    for event in response:
        if event.is_final_response():
            print(f"  Expert response: {event.content.parts[0].text[:200]}")
