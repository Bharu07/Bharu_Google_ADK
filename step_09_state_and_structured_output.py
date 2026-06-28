# ============================================================
# GOOGLE ADK — STEP 09: State Management & Structured Output
# Goal: Share data between agents via state + force JSON output
# Run:  python step_09_state_and_structured_output.py
# ============================================================

# ── KEY CONCEPTS ─────────────────────────────────────────────────────────────
#
# 1. SESSION STATE — A key-value store attached to each session.
#    - Agents can READ and WRITE to state.
#    - Use output_key to auto-save an agent's final response to state.
#    - Downstream agents (in a SequentialAgent) can read it.
#
# 2. STRUCTURED OUTPUT (output_schema) — Force the LLM to return JSON
#    matching a specific Pydantic schema.
#    - Equivalent to LangChain's .with_structured_output()
#    - ADK enforces it at the model level (response_schema).
#
# LANGCHAIN COMPARISON:
#   LangChain:  model.with_structured_output(MySchema)
#   ADK:        Agent(output_schema=MySchema, ...)
#
#   LangChain:  state["key"] = value  (in LangGraph)
#   ADK:        output_key="key" → agent's reply auto-saved to session state
#               OR: use tool_context.state["key"] = value inside tools

# ── IMPORTS ──────────────────────────────────────────────────────────────────

import os
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google.adk import Agent
from google.adk.agents import SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

load_dotenv()
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

session_service = InMemorySessionService()
APP_NAME = "state_demo"
USER_ID = "user_1"


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: output_key — Auto-save agent output to session state
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PART 1: output_key (Agent → State → Next Agent)")
print("=" * 60)
print("SequentialAgent: researcher saves to state → writer reads from state")

# ── HOW output_key WORKS ─────────────────────────────────────────────────────
#
# When you set output_key="research_results" on an agent:
#   1. Agent generates its response normally
#   2. ADK automatically saves that response text to session state:
#      session.state["research_results"] = "the agent's full text response"
#   3. The NEXT agent in a SequentialAgent can read it via {research_results}
#      placeholder in its instruction!
#
# This is HOW agents share data in a pipeline.

researcher = Agent(
    name="researcher",
    model=f"azure/{AZURE_DEPLOYMENT}",
    instruction="""Research the given topic and provide exactly 3 key facts.
    Format as a numbered list. Be concise (one sentence per fact).""",
    output_key="research_results",
    # ^^^ Agent's response automatically stored in state["research_results"]
)

writer = Agent(
    name="writer",
    model=f"azure/{AZURE_DEPLOYMENT}",
    instruction="""You are a writer. Using the research below, write a short
    engaging paragraph (2-3 sentences) that combines the facts naturally.

    RESEARCH FACTS:
    {research_results}
    """,
    # ^^^ {research_results} is replaced with state["research_results"] at runtime!
    # This is how the writer reads the researcher's output.
    output_key="final_article",
)

pipeline = SequentialAgent(
    name="content_pipeline",
    sub_agents=[researcher, writer],
    # Flow: user question → researcher (saves to state) → writer (reads from state)
)


async def main():

    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner = Runner(agent=pipeline, app_name=APP_NAME, session_service=session_service)

    msg = Content(role="user", parts=[Part(text="Tell me about quantum computing")])
    response = runner.run(session_id=session.id, user_id=USER_ID, new_message=msg)

    print("\nPipeline result:")
    for event in response:
        if event.is_final_response():
            print(f"  [{event.author}]: {event.content.parts[0].text[:200]}")

    # You can also inspect the session state directly:
    print("\n  State keys saved:")
    print(f"    'research_results' → {str(session.state.get('research_results', 'N/A'))[:80]}...")
    print(f"    'final_article' → {str(session.state.get('final_article', 'N/A'))[:80]}...")


    # ══════════════════════════════════════════════════════════════════════════════
    # PART 2: Writing to state from TOOLS (tool_context.state)
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("PART 2: Writing State from Tools (tool_context)")
    print("=" * 60)


    def lookup_user(user_id: str, tool_context) -> str:
        """Look up user information by ID.

        Args:
            user_id: The user's ID to look up.
            tool_context: Provided automatically by ADK — gives access to state.
        """
        # Simulated database
        users = {
            "U001": {"name": "Alice", "plan": "premium", "tickets": 3},
            "U002": {"name": "Bob", "plan": "free", "tickets": 0},
        }

        user = users.get(user_id)
        if user:
            # Save user info to session state for other tools/agents to use!
            tool_context.state["current_user_name"] = user["name"]
            tool_context.state["current_user_plan"] = user["plan"]
            tool_context.state["current_user_tickets"] = user["tickets"]
            return f"Found user: {user['name']} (Plan: {user['plan']}, Open tickets: {user['tickets']})"
        return f"User {user_id} not found."


    def check_eligibility(feature: str, tool_context) -> str:
        """Check if the current user is eligible for a feature.

        Args:
            feature: The feature to check eligibility for.
            tool_context: Provided automatically by ADK.
        """
        # Read state that was written by lookup_user!
        plan = tool_context.state.get("current_user_plan", "unknown")
        name = tool_context.state.get("current_user_name", "Unknown")

        premium_features = ["priority support", "analytics", "api access"]

        if plan == "premium":
            return f"{name} has premium — eligible for all features including {feature}."
        elif feature.lower() in premium_features:
            return f"{name} is on free plan — NOT eligible for {feature}. Upgrade needed."
        else:
            return f"{name} is eligible for {feature} on all plans."


    # NOTE: When a tool parameter is named tool_context, ADK auto-injects it!
    # You don't pass it manually — ADK provides it. The LLM doesn't see it.

    support_agent = Agent(
        name="support_agent",
        model=f"azure/{AZURE_DEPLOYMENT}",
        instruction="""You are a support agent. When asked about a user:
        1. First look up the user with lookup_user
        2. Then check their eligibility if asked
        Be concise and helpful.""",
        tools=[lookup_user, check_eligibility],
    )

    session2 = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner2 = Runner(agent=support_agent, app_name=APP_NAME, session_service=session_service)

    print("\nTest: 'Look up user U001 and check if they can use priority support'")
    msg = Content(role="user", parts=[Part(text=
        "Look up user U001 and check if they can use priority support"
    )])
    response = runner2.run(session_id=session2.id, user_id=USER_ID, new_message=msg)
    for event in response:
        if event.is_final_response():
            print(f"  Response: {event.content.parts[0].text[:200]}")

    # Check state was written by the tools:
    print(f"\n  State after tool calls:")
    print(f"    current_user_name = {session2.state.get('current_user_name', 'N/A')}")
    print(f"    current_user_plan = {session2.state.get('current_user_plan', 'N/A')}")


    # ══════════════════════════════════════════════════════════════════════════════
    # PART 3: output_schema — Structured JSON Output
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("PART 3: output_schema (Structured Output)")
    print("=" * 60)

    # ── HOW output_schema WORKS ──────────────────────────────────────────────────
    #
    # Pass a Pydantic model to output_schema:
    #   - The LLM is FORCED to return JSON matching that schema.
    #   - No more parsing/hoping the LLM returns valid JSON!
    #
    # LangChain equivalent:
    #   model.with_structured_output(MovieReview)
    #
    # ADK equivalent:
    #   Agent(output_schema=MovieReview, ...)

    class MovieReview(BaseModel):
        """A structured movie review."""
        title: str = Field(description="The movie title")
        rating: float = Field(description="Rating out of 10")
        genre: str = Field(description="Primary genre")
        summary: str = Field(description="One-sentence summary")
        recommended: bool = Field(description="Whether you recommend it")


    structured_agent = Agent(
        name="movie_reviewer",
        model=f"azure/{AZURE_DEPLOYMENT}",
        instruction="You are a movie critic. When given a movie name, provide a structured review.",
        output_schema=MovieReview,
        # ^^^ Forces the LLM to return JSON matching MovieReview schema!
        # The response will be a JSON string that can be parsed into MovieReview.
    )

    session3 = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner3 = Runner(agent=structured_agent, app_name=APP_NAME, session_service=session_service)

    print("\nTest: 'Review the movie Inception'")
    msg = Content(role="user", parts=[Part(text="Review the movie Inception")])
    response = runner3.run(session_id=session3.id, user_id=USER_ID, new_message=msg)
    for event in response:
        if event.is_final_response():
            raw_json = event.content.parts[0].text
            print(f"  Raw JSON response:\n  {raw_json[:300]}")

            # Parse it into a Pydantic model:
            import json
            try:
                data = json.loads(raw_json)
                review = MovieReview(**data)
                print(f"\n  Parsed structured output:")
                print(f"    Title: {review.title}")
                print(f"    Rating: {review.rating}/10")
                print(f"    Genre: {review.genre}")
                print(f"    Summary: {review.summary}")
                print(f"    Recommended: {'Yes' if review.recommended else 'No'}")
            except (json.JSONDecodeError, Exception) as e:
                print(f"  (Parse note: {e})")


    # ══════════════════════════════════════════════════════════════════════════════
    # PART 4: Initial State (Pre-loading session state)
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("PART 4: Pre-loaded Session State")
    print("=" * 60)

    # You can pre-load state when creating a session!
    # Useful for: user preferences, context from external systems, etc.

    personalized_agent = Agent(
        name="personalized_agent",
        model=f"azure/{AZURE_DEPLOYMENT}",
        instruction="""You are a personal assistant for {user_name}.
        Their preferred language is {preferred_language}.
        Greet them by name and respond in their preferred language.
        Be concise.""",
        # ^^^ Placeholders are filled from session state!
    )

    # Pre-load state when creating the session
    session4 = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        state={
            "user_name": "Bharath",
            "preferred_language": "English",
            "timezone": "IST",
        },
        # ^^^ This state is available immediately — no need to set it later!
    )
    runner4 = Runner(agent=personalized_agent, app_name=APP_NAME, session_service=session_service)

    print("\nTest: Pre-loaded state → personalized greeting")
    msg = Content(role="user", parts=[Part(text="Hello!")])
    response = runner4.run(session_id=session4.id, user_id=USER_ID, new_message=msg)
    for event in response:
        if event.is_final_response():
            print(f"  Response: {event.content.parts[0].text[:150]}")


    # ══════════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("""
    STATE & STRUCTURED OUTPUT SUMMARY:
    ─────────────────────────────────────────────────────────────
    1. output_key="key"
       → Agent's text response auto-saved to state["key"]
       → Next agent reads it via {key} in its instruction

    2. tool_context.state["key"] = value
       → Tools can read/write session state
       → tool_context is auto-injected (name the parameter 'tool_context')

    3. output_schema=PydanticModel
       → Force LLM to output JSON matching the schema
       → Parse with json.loads() + Model(**data)

    4. state={...} in create_session()
       → Pre-load state before agent runs
       → Available immediately to {placeholders} in instructions

    5. {placeholder} in instruction strings
       → Auto-replaced with state["placeholder"] at runtime
    """)



asyncio.run(main())

