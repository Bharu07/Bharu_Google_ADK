# ============================================================
# GOOGLE ADK — STEP 11: ADK 2.0 — Workflow API & Agent Modes
# Goal: Learn the new graph-based Workflow and Task/SingleTurn modes
# Run:  python step_11_workflow_and_modes.py
# ============================================================

# ── KEY CONCEPT ──────────────────────────────────────────────────────────────
#
# ADK 2.0 introduces TWO major new features:
#
# 1. WORKFLOW API — A graph-based execution engine (replaces SequentialAgent
#    and ParallelAgent for complex flows).
#    - Define edges: ("START", agent_a, agent_b) → A runs then B
#    - Supports routing, fan-out/fan-in, loops, retry, human-in-the-loop
#
# 2. AGENT MODES (Task API) — Structured agent-to-agent delegation:
#    - mode="chat"        → Standard sub-agent (transfers control)
#    - mode="single_turn" → Sub-agent answers once and returns (like a tool)
#    - mode="task"        → Sub-agent runs a multi-turn task, then returns
#
# COMPARISON:
#   ADK 1.x:  SequentialAgent([A, B, C])      — agents in order
#   ADK 2.0:  Workflow(edges=[("START", A, B, C)])  — graph with edges
#
#   ADK 1.x:  sub_agents=[agent]               — LLM chooses when to delegate
#   ADK 2.0:  sub_agents=[agent] + mode="task"  — structured delegation
#
# LANGCHAIN EQUIVALENT:
#   This is closest to LangGraph's StateGraph with nodes and edges.
#   But ADK's syntax is much simpler!

# ── IMPORTS ──────────────────────────────────────────────────────────────────

import os
import asyncio
from dotenv import load_dotenv
from google.adk import Agent

# NEW IN ADK 2.0:
from google.adk import Workflow
# Workflow is the new graph-based orchestrator.
# It replaces SequentialAgent/ParallelAgent for complex flows.

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

load_dotenv()
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

session_service = InMemorySessionService()
APP_NAME = "workflow_demo"
USER_ID = "user_1"


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: BASIC WORKFLOW (Sequential Graph)
# Equivalent to SequentialAgent but using the Workflow API.
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PART 1: Basic Workflow (Sequential Graph)")
print("=" * 60)

# ── HOW WORKFLOW EDGES WORK ──────────────────────────────────────────────────
#
# edges=[("START", agent_a, agent_b)]
#
# This means: START → agent_a → agent_b → END
#
# Multiple agents in one tuple = sequential execution.
# "START" is a special keyword meaning "begin here".
#
# Think of it like a pipeline:
#   Input → agent_a processes → agent_b processes → Output

researcher = Agent(
    name="researcher",
    model=f"azure/{AZURE_DEPLOYMENT}",
    instruction="""Research the given topic. Provide 3 key facts as a numbered list.
    Keep each fact to one sentence.""",
    output_key="research_facts",
    # ^^^ Saves output to state["research_facts"] for the next agent
)

writer = Agent(
    name="writer",
    model=f"azure/{AZURE_DEPLOYMENT}",
    instruction="""Write a short paragraph (2-3 sentences) using these facts:
    {research_facts}
    Make it engaging and readable.""",
    output_key="article",
)

# WORKFLOW: START → researcher → writer
content_workflow = Workflow(
    name="content_workflow",
    edges=[("START", researcher, writer)],
    # ^^^ Simple sequential: researcher runs first, then writer.
    # This is equivalent to SequentialAgent([researcher, writer])
    # but using the new graph-based API.
)


async def main():

    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner = Runner(agent=content_workflow, app_name=APP_NAME, session_service=session_service)

    print("\nWorkflow: START → researcher → writer")
    msg = Content(role="user", parts=[Part(text="Tell me about black holes")])
    response = runner.run(session_id=session.id, user_id=USER_ID, new_message=msg)
    for event in response:
        if event.is_final_response():
            print(f"  [{event.author}]: {event.content.parts[0].text[:200]}...")


    # ══════════════════════════════════════════════════════════════════════════════
    # PART 2: BRANCHING WORKFLOW (Fan-out / Parallel)
    # Multiple paths from one node = parallel execution.
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("PART 2: Branching Workflow (Fan-out)")
    print("=" * 60)

    # ── FAN-OUT PATTERN ──────────────────────────────────────────────────────────
    #
    # edges=[
    #     ("START", analyst_a),
    #     ("START", analyst_b),
    #     (analyst_a, summarizer),
    #     (analyst_b, summarizer),
    # ]
    #
    # This creates:
    #       ┌─→ analyst_a ─┐
    # START─┤              ├─→ summarizer
    #       └─→ analyst_b ─┘
    #
    # analyst_a and analyst_b run IN PARALLEL!
    # summarizer waits for BOTH to finish, then runs.

    pros_analyst = Agent(
        name="pros_analyst",
        model=f"azure/{AZURE_DEPLOYMENT}",
        instruction="""Analyze the given topic and list 3 PROS/ADVANTAGES.
        Format: numbered list, one sentence each.""",
        output_key="pros",
    )

    cons_analyst = Agent(
        name="cons_analyst",
        model=f"azure/{AZURE_DEPLOYMENT}",
        instruction="""Analyze the given topic and list 3 CONS/DISADVANTAGES.
        Format: numbered list, one sentence each.""",
        output_key="cons",
    )

    summarizer = Agent(
        name="summarizer",
        model=f"azure/{AZURE_DEPLOYMENT}",
        instruction="""Summarize this analysis into a balanced 2-sentence conclusion.

        PROS:
        {pros}

        CONS:
        {cons}
        """,
    )

    # Fan-out: START splits to pros + cons, then both feed into summarizer
    analysis_workflow = Workflow(
        name="analysis_workflow",
        edges=[
            ("START", pros_analyst),
            ("START", cons_analyst),
            (pros_analyst, summarizer),
            (cons_analyst, summarizer),
        ],
        # pros_analyst and cons_analyst run in PARALLEL
        # summarizer runs only after BOTH complete
    )

    session2 = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner2 = Runner(agent=analysis_workflow, app_name=APP_NAME, session_service=session_service)

    print("\nWorkflow: START → [pros + cons in parallel] → summarizer")
    msg = Content(role="user", parts=[Part(text="Remote work")])
    response = runner2.run(session_id=session2.id, user_id=USER_ID, new_message=msg)
    for event in response:
        if event.is_final_response():
            print(f"  [{event.author}]: {event.content.parts[0].text[:200]}")

    # Check state
    print(f"\n  State['pros']: {str(session2.state.get('pros', 'N/A'))[:80]}...")
    print(f"  State['cons']: {str(session2.state.get('cons', 'N/A'))[:80]}...")


    # ══════════════════════════════════════════════════════════════════════════════
    # PART 3: AGENT MODES — single_turn (Agent-as-Tool)
    # Sub-agent runs once and returns (like calling a function).
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("PART 3: Agent Modes — single_turn")
    print("=" * 60)

    # ── AGENT MODES EXPLAINED ────────────────────────────────────────────────────
    #
    # mode="chat" (default for sub-agents):
    #   - Control TRANSFERS to the sub-agent.
    #   - Sub-agent can have multi-turn conversation with the user.
    #   - Like handing the phone to a specialist.
    #
    # mode="single_turn":
    #   - Sub-agent runs ONCE and returns result to parent.
    #   - The parent continues the conversation.
    #   - Like asking a colleague a quick question.
    #   - The sub-agent is exposed as a TOOL to the parent agent!
    #
    # mode="task":
    #   - Sub-agent runs a structured task (potentially multi-turn).
    #   - Reports back to parent when done.
    #   - Like delegating a project to a team member.

    # A single_turn agent — acts like a tool for the parent
    translator = Agent(
        name="translate_to_spanish",
        model=f"azure/{AZURE_DEPLOYMENT}",
        instruction="""Translate the given text to Spanish.
        Return ONLY the translation, nothing else.""",
        mode="single_turn",
        # ^^^ This agent runs once and returns.
        # The parent sees it as a tool it can call!
        description="Translates text to Spanish.",
    )

    sentiment_analyzer = Agent(
        name="analyze_sentiment",
        model=f"azure/{AZURE_DEPLOYMENT}",
        instruction="""Analyze the sentiment of the given text.
        Return exactly one word: POSITIVE, NEGATIVE, or NEUTRAL.""",
        mode="single_turn",
        description="Analyzes text sentiment (returns POSITIVE/NEGATIVE/NEUTRAL).",
    )

    # Parent agent that uses single_turn agents like tools
    coordinator = Agent(
        name="coordinator",
        model=f"azure/{AZURE_DEPLOYMENT}",
        instruction="""You are a text processing assistant.
        You can translate text and analyze sentiment.
        Use the appropriate sub-agent for each task.
        Summarize results clearly.""",
        sub_agents=[translator, sentiment_analyzer],
        # ^^^ Because translator and sentiment_analyzer have mode="single_turn",
        # they appear as TOOLS to the coordinator.
        # The coordinator calls them like functions and gets results back.
    )

    session3 = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner3 = Runner(agent=coordinator, app_name=APP_NAME, session_service=session_service)

    print("\nTest: 'Translate \"I love programming\" to Spanish and analyze its sentiment'")
    msg = Content(role="user", parts=[Part(text=
        "Translate 'I love programming' to Spanish and also analyze its sentiment."
    )])
    response = runner3.run(session_id=session3.id, user_id=USER_ID, new_message=msg)
    for event in response:
        if event.is_final_response():
            print(f"  Response: {event.content.parts[0].text[:200]}")


    # ══════════════════════════════════════════════════════════════════════════════
    # PART 4: AGENT MODES — task (Structured Delegation)
    # Sub-agent chats with user to complete a task, then returns.
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("PART 4: Agent Modes — task")
    print("=" * 60)

    # ── TASK MODE ────────────────────────────────────────────────────────────────
    #
    # mode="task" agents:
    #   - The parent delegates a TASK to the sub-agent.
    #   - The sub-agent works on it (possibly asking the user clarifying questions).
    #   - When done, it calls finish_task() to report back to the parent.
    #   - The parent resumes control.
    #
    # Think of it as: "Hey sub-agent, go collect the user's shipping address,
    # then come back to me with the result."

    order_collector = Agent(
        name="collect_order_details",
        model=f"azure/{AZURE_DEPLOYMENT}",
        instruction="""You are an order details collector.
        Your task: Get the user's preferred pizza topping and size (small/medium/large).
        Once you have both pieces of information, finish your task by summarizing:
        "Order: [size] pizza with [topping]"
        Be conversational but focused.""",
        mode="task",
        description="Collects pizza order details (topping and size) from the user.",
        # NOTE: In a real task agent, it would call finish_task() when done.
        # The task agent can have multi-turn conversation to gather info.
    )

    pizza_bot = Agent(
        name="pizza_bot",
        model=f"azure/{AZURE_DEPLOYMENT}",
        instruction="""You are a pizza ordering bot.
        When a user wants to order, delegate to collect_order_details to gather their preferences.
        After receiving the order details, confirm the order to the user.""",
        sub_agents=[order_collector],
    )

    session4 = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner4 = Runner(agent=pizza_bot, app_name=APP_NAME, session_service=session_service)

    print("\nTest: Pizza ordering with task delegation")
    msg = Content(role="user", parts=[Part(text="I'd like to order a pizza please")])
    response = runner4.run(session_id=session4.id, user_id=USER_ID, new_message=msg)
    for event in response:
        if event.is_final_response():
            print(f"  Response: {event.content.parts[0].text[:200]}")


    # ══════════════════════════════════════════════════════════════════════════════
    # PART 5: TRANSFER CONTROLS
    # Control how agents can navigate the agent tree.
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("PART 5: Transfer Controls")
    print("=" * 60)

    # ── TRANSFER CONTROLS ────────────────────────────────────────────────────────
    #
    # disallow_transfer_to_parent=True
    #   → Agent CANNOT transfer back to parent.
    #   → Prevents "stuck" loops where agents bounce back and forth.
    #   → After it responds, control auto-returns to parent next turn.
    #
    # disallow_transfer_to_peers=True
    #   → Agent CANNOT transfer to sibling agents.
    #   → Forces it to handle the question itself or escalate to parent.

    # A focused agent that can't escape to parent or siblings
    focused_helper = Agent(
        name="focused_helper",
        model=f"azure/{AZURE_DEPLOYMENT}",
        instruction="""You ONLY answer questions about Python programming.
        For anything else, say 'I can only help with Python questions.'""",
        description="Answers Python programming questions only.",
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        # ^^^ This agent is "locked in" — it must handle whatever it receives.
        # It can't pass the buck to other agents.
    )

    general_agent = Agent(
        name="general_agent",
        model=f"azure/{AZURE_DEPLOYMENT}",
        instruction="""You are a general assistant. Route Python questions to focused_helper.
        Handle everything else yourself. Be concise.""",
        sub_agents=[focused_helper],
    )

    session5 = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner5 = Runner(agent=general_agent, app_name=APP_NAME, session_service=session_service)

    print("\nTest: Route to locked sub-agent")
    msg = Content(role="user", parts=[Part(text="How do I create a list in Python?")])
    response = runner5.run(session_id=session5.id, user_id=USER_ID, new_message=msg)
    for event in response:
        if event.is_final_response():
            print(f"  Response: {event.content.parts[0].text[:150]}")


    # ══════════════════════════════════════════════════════════════════════════════
    # COMPARISON: OLD vs NEW API
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("""
    ADK 1.x vs 2.0 COMPARISON:
    ─────────────────────────────────────────────────────────────
    SEQUENTIAL EXECUTION:
      1.x: SequentialAgent(sub_agents=[A, B, C])
      2.0: Workflow(edges=[("START", A, B, C)])

    PARALLEL EXECUTION:
      1.x: ParallelAgent(sub_agents=[A, B, C])
      2.0: Workflow(edges=[("START", A), ("START", B), ("START", C),
                           (A, merger), (B, merger), (C, merger)])

    FAN-OUT + FAN-IN:
      1.x: Not easily possible without custom code.
      2.0: Workflow(edges=[("START", A), ("START", B), (A, C), (B, C)])

    SUB-AGENT DELEGATION:
      1.x: sub_agents=[agent]  → LLM decides when to delegate
      2.0: mode="chat" (same), mode="single_turn" (like a tool),
           mode="task" (structured delegation)

    TRANSFER CONTROLS:
      1.x: Not available
      2.0: disallow_transfer_to_parent, disallow_transfer_to_peers

    KEY DIFFERENCES:
      • Workflow gives you explicit graph control (like LangGraph but simpler)
      • Agent modes give you structured delegation patterns
      • Both old SequentialAgent/ParallelAgent still work in 2.0
      • Workflow is preferred for new code (more flexible)
    """)



asyncio.run(main())
