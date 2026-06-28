# ============================================================
# GOOGLE ADK — STEP 01: Your First Agent (Azure OpenAI)
# Goal: Build a basic agent using Azure OpenAI model via ADK
# Run:  python step_01_basic_agent.py
# ============================================================

# ── INSTALL (run once in terminal) ───────────────────────────────────────────
# pip install google-adk python-dotenv litellm openai

# ── IMPORTS ──────────────────────────────────────────────────────────────────

import os
import asyncio
from dotenv import load_dotenv
# load_dotenv reads your .env file and loads variables into the environment.
# Same as you used in LangChain!

from google.adk import Agent
# Agent is the CORE class in ADK — equivalent to create_agent() in LangChain.
# It wraps a model + instructions + tools into one object.

from google.adk.runners import Runner
# Runner is the EXECUTOR — it actually runs the agent.
# In LangChain you called agent.invoke() directly.
# In ADK you create a Runner and call runner.run().
# Why? Because Runner handles sessions, streaming, and multi-turn automatically.

from google.adk.sessions import InMemorySessionService
# InMemorySessionService stores conversation history in memory (RAM).
# It's like keeping a list of messages in LangChain, but ADK manages it for you.
# "InMemory" means it resets when your program restarts.

from google.genai.types import Content, Part
# Content and Part are how ADK represents messages.
# Content = a message (like HumanMessage or AIMessage in LangChain)
# Part = the actual text inside the message
#
# LangChain: HumanMessage("Hello")
# ADK:       Content(role="user", parts=[Part(text="Hello")])


# ── SETUP (same pattern as your LangChain project) ────────────────────────────

load_dotenv()
# Reads .env file in the current directory and loads variables.
# LiteLLM reads AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION
# directly from the environment — no remapping needed.

AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


# ── HOW ADK USES AZURE OPENAI ────────────────────────────────────────────────
#
# ADK natively supports Gemini. For OTHER models (Azure OpenAI, OpenAI, Anthropic),
# ADK uses LiteLLM under the hood.
#
# Model string format: "azure/<your-deployment-name>"
#
# LangChain:  init_chat_model("azure_openai:gpt-4o", azure_deployment="gpt-4o")
# ADK:        Agent(model="azure/gpt-4o", ...)
#
# LiteLLM reads AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION from environment.


# ── STEP 1: CREATE AN AGENT ──────────────────────────────────────────────────

# LANGCHAIN WAY (what you already know):
#   model = init_chat_model("azure_openai:gpt-4o", azure_deployment=DEPLOYMENT)
#   agent = create_agent(model, tools=[...], system_prompt="...")
#
# ADK WAY (new):
#   agent = Agent(name="...", model="azure/gpt-4o", instruction="...", tools=[...])

agent = Agent(
    name="my_first_agent",
    # Every agent needs a unique name.
    # Used for logging, debugging, and multi-agent routing.
    # LangChain doesn't require this — ADK does.

    model=f"azure/{AZURE_DEPLOYMENT}",
    # Format: "azure/<deployment-name>"
    # "litellm/" prefix tells ADK to use LiteLLM for this model.
    # "azure/" tells LiteLLM it's an Azure OpenAI deployment.
    # The deployment name (gpt-4o) comes from your .env file.
    #
    # OTHER OPTIONS:
    # Google Gemini:    model="gemini-2.0-flash"
    # Regular OpenAI:   model="litellm/openai/gpt-4o"
    # Anthropic:        model="litellm/anthropic/claude-3-sonnet"

    instruction="You are a helpful assistant. Answer questions clearly and concisely.",
    # This is the SYSTEM PROMPT — same concept as in LangChain.
    # LangChain called it "system_prompt". ADK calls it "instruction".
    # It sets the agent's behavior and personality.

    description="A general-purpose helpful assistant.",
    # Description is used when this agent is a sub-agent.
    # The parent agent reads this to decide when to delegate to this agent.
)


# ── STEP 2: CREATE A SESSION SERVICE ─────────────────────────────────────────

# In LangChain, you managed conversation history manually:
#   messages = [HumanMessage("hi"), AIMessage("hello"), HumanMessage("bye")]
#
# In ADK, the Session handles this automatically.

session_service = InMemorySessionService()
# Creates a session manager that stores conversations in memory.

APP_NAME = "my_first_app"
USER_ID = "user_1"

# NOTE: In ADK 2.x, create_session() is async.
# So we wrap our main logic in an async function.


# ── STEP 3: CREATE A RUNNER ──────────────────────────────────────────────────

# In LangChain: result = agent.invoke({"messages": [...]})
# In ADK: runner.run(session_id, user_id, new_message)

runner = Runner(
    agent=agent,
    app_name=APP_NAME,
    session_service=session_service,
)


# ── STEP 4: SEND A MESSAGE AND GET A RESPONSE ────────────────────────────────

async def main():
    # create_session is async in ADK 2.x
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
    )
    # Creates a new conversation session.
    # app_name = identifies your application
    # user_id = identifies the user
    # Returns a session object with a unique session.id

    print("=" * 55)
    print("GOOGLE ADK — STEP 01: Basic Agent (Azure OpenAI)")
    print("=" * 55)

    user_message = Content(
        role="user",
        parts=[Part(text="What is artificial intelligence? Explain in 2 sentences.")]
    )
    # Content = a message object (like HumanMessage in LangChain)
    # role="user" = from the human
    # role="model" = from the AI

    response = runner.run(
        session_id=session.id,
        user_id=USER_ID,
        new_message=user_message,
    )

    print("\nAgent's response:")
    for event in response:
        if event.is_final_response():
            print(event.content.parts[0].text)


    # ── STEP 5: MULTI-TURN (session remembers automatically) ─────────────────

    print("\n" + "=" * 55)
    print("FOLLOW-UP (session remembers context)")
    print("=" * 55)

    followup = Content(
        role="user",
        parts=[Part(text="Can you give me an example of it in daily life?")]
    )
    # "it" refers to AI from the first question — session remembers!

    response = runner.run(
        session_id=session.id,  # SAME session = same conversation
        user_id=USER_ID,
        new_message=followup,
    )

    print("\nAgent's response:")
    for event in response:
        if event.is_final_response():
            print(event.content.parts[0].text)


# Run the async main function
asyncio.run(main())


# ── WHAT WE LEARNED ──────────────────────────────────────────────────────────
#
# 1. For Azure OpenAI: model="azure/<deployment-name>"
# 2. Set AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION in environment
# 3. Agent(name, model, instruction) creates an agent
# 4. InMemorySessionService() manages conversation history automatically
# 5. Runner ties agent + session together
# 6. runner.run() sends a message (like agent.invoke() in LangChain)
# 7. Sessions remember context across turns automatically
#
# NEXT: step_02_tools.py — Adding tools