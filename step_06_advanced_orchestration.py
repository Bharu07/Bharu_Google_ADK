# ============================================================
# GOOGLE ADK — STEP 06: Advanced Orchestration Patterns
# Goal: Learn SequentialAgent, ParallelAgent, LoopAgent
# Run:  python step_06_advanced_orchestration.py
# ============================================================

# ── KEY CONCEPT ──────────────────────────────────────────────────────────────
#
# In Step 05, we built a simple orchestrator (LLM-based routing).
# ADK also provides STRUCTURAL orchestration patterns:
#
#   1. SequentialAgent → Runs agents in ORDER (A → B → C)
#   2. ParallelAgent   → Runs agents AT THE SAME TIME
#   3. LoopAgent       → Runs agents in a LOOP until done
#
# LangChain has NOTHING equivalent to these built-in.
# In LangChain, you'd code these manually with LangGraph.

# ── IMPORTS ──────────────────────────────────────────────────────────────────

import os
import asyncio
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.agents import SequentialAgent, ParallelAgent, LoopAgent

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

load_dotenv()
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

session_service = InMemorySessionService()
APP_NAME = "orchestration_demo"
USER_ID = "user_1"


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 1: SEQUENTIAL AGENT (Pipeline)
# ══════════════════════════════════════════════════════════════════════════════

researcher = Agent(
    name="researcher",
    model=f"azure/{AZURE_DEPLOYMENT}",
    instruction="""You are a researcher. When given a topic, provide 3 key facts about it.
    Format: numbered list of facts. Keep each fact to one sentence.""",
    description="Researches topics and provides key facts.",
)

writer = Agent(
    name="writer",
    model=f"azure/{AZURE_DEPLOYMENT}",
    instruction="""You are a writer. Take the facts provided by the researcher
    and write a short, engaging paragraph (3-4 sentences) combining them.
    Make it readable and interesting.""",
    description="Writes engaging content from research facts.",
)

reviewer = Agent(
    name="reviewer",
    model=f"azure/{AZURE_DEPLOYMENT}",
    instruction="""You are an editor/reviewer. Review the written paragraph and:
    1. Fix any grammar issues
    2. Suggest one improvement
    3. Give a final polished version.
    Keep your output concise.""",
    description="Reviews and polishes written content.",
)

content_pipeline = SequentialAgent(
    name="content_pipeline",
    sub_agents=[researcher, writer, reviewer],
)


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 2: PARALLEL AGENT (Concurrent)
# ══════════════════════════════════════════════════════════════════════════════

weather_reporter = Agent(
    name="weather_reporter",
    model=f"azure/{AZURE_DEPLOYMENT}",
    instruction="Provide a brief weather summary for New York. Make up realistic data. One sentence.",
    description="Reports weather.",
)

news_reporter = Agent(
    name="news_reporter",
    model=f"azure/{AZURE_DEPLOYMENT}",
    instruction="Provide one brief fictional headline for today. One sentence.",
    description="Reports news.",
)

trivia_agent = Agent(
    name="trivia_agent",
    model=f"azure/{AZURE_DEPLOYMENT}",
    instruction="Provide one interesting science fact. One sentence.",
    description="Provides trivia.",
)

morning_briefing = ParallelAgent(
    name="morning_briefing",
    sub_agents=[weather_reporter, news_reporter, trivia_agent],
)


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 3: LOOP AGENT (Iterative Refinement)
# ══════════════════════════════════════════════════════════════════════════════

drafter = Agent(
    name="drafter",
    model=f"azure/{AZURE_DEPLOYMENT}",
    instruction="""Write or improve a haiku about technology.
    If previous feedback exists, incorporate it.
    Output ONLY the haiku (3 lines).""",
    description="Writes and improves haiku.",
)

critic = Agent(
    name="critic",
    model=f"azure/{AZURE_DEPLOYMENT}",
    instruction="""Critique the haiku. Check:
    1. Does it follow 5-7-5 syllable pattern?
    2. Is it about technology?
    3. Is it evocative/beautiful?
    Give one specific suggestion for improvement.
    If it's good enough, say 'APPROVED' to end the loop.""",
    description="Critiques haiku and approves when ready.",
)

haiku_refiner = LoopAgent(
    name="haiku_refiner",
    sub_agents=[drafter, critic],
    max_iterations=3,
)


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    # --- PATTERN 1: Sequential ---
    print("=" * 60)
    print("PATTERN 1: SequentialAgent (Pipeline)")
    print("=" * 60)
    print("Agents run in ORDER: researcher → writer → reviewer")

    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner = Runner(agent=content_pipeline, app_name=APP_NAME, session_service=session_service)
    message = Content(role="user", parts=[Part(text="Write about artificial intelligence")])
    response = runner.run(session_id=session.id, user_id=USER_ID, new_message=message)

    print("\nPipeline result:")
    for event in response:
        if event.is_final_response():
            print(f"  [{event.author}]: {event.content.parts[0].text[:200]}...")

    # --- PATTERN 2: Parallel ---
    print("\n" + "=" * 60)
    print("PATTERN 2: ParallelAgent (Concurrent)")
    print("=" * 60)
    print("Agents run SIMULTANEOUSLY: weather + news + trivia")

    session2 = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner2 = Runner(agent=morning_briefing, app_name=APP_NAME, session_service=session_service)
    message = Content(role="user", parts=[Part(text="Give me my morning briefing")])
    response = runner2.run(session_id=session2.id, user_id=USER_ID, new_message=message)

    print("\nParallel results (all fetched simultaneously):")
    for event in response:
        if event.content and event.content.parts and event.content.parts[0].text:
            print(f"  [{event.author}]: {event.content.parts[0].text[:100]}")

    # --- PATTERN 3: Loop ---
    print("\n" + "=" * 60)
    print("PATTERN 3: LoopAgent (Iterative)")
    print("=" * 60)
    print("Agents run in a LOOP: draft → critique → improve (repeat)")

    session3 = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner3 = Runner(agent=haiku_refiner, app_name=APP_NAME, session_service=session_service)
    message = Content(role="user", parts=[Part(text="Create a haiku about coding")])
    response = runner3.run(session_id=session3.id, user_id=USER_ID, new_message=message)

    print("\nLoop iterations:")
    for event in response:
        if event.content and event.content.parts and event.content.parts[0].text:
            text = event.content.parts[0].text[:150]
            print(f"  [{event.author}]: {text}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("""
SUMMARY:
  SequentialAgent → Pipeline (A → B → C)
  ParallelAgent   → Concurrent (A + B + C at same time)
  LoopAgent       → Iterative (A → B → A → B until done)
  Agent + sub_agents → LLM-based smart routing
""")


asyncio.run(main())