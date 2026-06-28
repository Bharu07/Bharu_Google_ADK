# ============================================================
# GOOGLE ADK — STEP 02: Tools (Giving your agent abilities)
# Goal: Learn how to create tools — just plain Python functions!
# Run:  python step_02_tools.py
# ============================================================

# ── KEY DIFFERENCE FROM LANGCHAIN ─────────────────────────────────────────────
#
# LANGCHAIN:
#   @tool                              ← decorator REQUIRED
#   def get_weather(location: str) -> str:
#       """Get the weather."""
#       return "Sunny"
#
# ADK:
#   def get_weather(location: str) -> str:   ← NO decorator needed!
#       """Get the weather."""
#       return "Sunny"
#
# ADK auto-generates the tool schema from type hints + docstring.
# Much simpler!

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


# ── DEFINE TOOLS (just plain Python functions!) ───────────────────────────────

def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city to get weather for.
    """
    # In LangChain, you'd use @tool decorator here.
    # In ADK, this is JUST a regular function.
    # ADK reads the docstring to understand what this tool does.
    # ADK reads the type hints (city: str) to know the parameters.

    weather_data = {
        "london": "Cloudy, 15°C, humidity 80%",
        "new york": "Sunny, 22°C, humidity 45%",
        "tokyo": "Rainy, 18°C, humidity 90%",
        "paris": "Partly cloudy, 17°C, humidity 65%",
        "mumbai": "Hot, 35°C, humidity 70%",
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"Weather in {city}: {weather_data[city_lower]}"
    return f"Sorry, I don't have weather data for {city}."


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Args:
        expression: A math expression like '2 + 2' or '10 * 5'.
    """
    try:
        # SECURITY NOTE: In production, use a safe math parser.
        allowed_chars = set("0123456789+-*/.() ")
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"The result of {expression} = {result}"
        else:
            return "Error: Expression contains invalid characters."
    except Exception as e:
        return f"Error computing: {e}"


def get_capital(country: str) -> str:
    """Get the capital city of a given country.

    Args:
        country: The name of the country.
    """
    capitals = {
        "france": "Paris",
        "japan": "Tokyo",
        "india": "New Delhi",
        "germany": "Berlin",
        "australia": "Canberra",
        "united states": "Washington, D.C.",
        "united kingdom": "London",
        "brazil": "Brasília",
    }
    country_lower = country.lower()
    if country_lower in capitals:
        return f"The capital of {country} is {capitals[country_lower]}."
    return f"Sorry, I don't know the capital of {country}."


# ── CREATE AGENT WITH TOOLS ──────────────────────────────────────────────────

agent = Agent(
    name="tool_agent",
    model=f"azure/{AZURE_DEPLOYMENT}",
    instruction="""You are a helpful assistant with access to tools.
    Use the available tools to answer questions accurately.
    If you don't need a tool, answer directly from your knowledge.
    Be concise in your responses.""",
    tools=[get_weather, calculate, get_capital],
    # Just pass the functions directly! No @tool decorator needed.
    # ADK automatically:
    #   1. Reads the function name → becomes the tool name
    #   2. Reads the docstring → becomes the tool description
    #   3. Reads type hints → becomes the parameter schema
)


# ── SETUP ─────────────────────────────────────────────────────────────────────

session_service = InMemorySessionService()
APP_NAME = "tool_demo"
USER_ID = "user_1"
runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)


# ── HELPER ────────────────────────────────────────────────────────────────────

async def main():
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)

    def ask_agent(question: str):
        """Send a question to the agent and print the response."""
        message = Content(role="user", parts=[Part(text=question)])
        response = runner.run(
            session_id=session.id,
            user_id=USER_ID,
            new_message=message,
        )
        for event in response:
            if event.is_final_response():
                print(f"Agent: {event.content.parts[0].text}")
                return

    # ── TEST ──────────────────────────────────────────────────────────────────────

    print("=" * 55)
    print("GOOGLE ADK — STEP 02: Tools (Azure OpenAI)")
    print("=" * 55)

    # Test 1: Weather tool
    print("\n--- Q1: Uses get_weather tool ---")
    print("User: What's the weather in Tokyo?")
    ask_agent("What's the weather in Tokyo?")

    # Test 2: Calculator tool
    print("\n--- Q2: Uses calculate tool ---")
    print("User: What is 145 multiplied by 23?")
    ask_agent("What is 145 multiplied by 23?")

    # Test 3: Capital tool
    print("\n--- Q3: Uses get_capital tool ---")
    print("User: What is the capital of India?")
    ask_agent("What is the capital of India?")

    # Test 4: Multi-tool (agent chains two tools automatically)
    print("\n--- Q4: Uses TWO tools (chaining) ---")
    print("User: What is the capital of Japan and what's the weather there?")
    ask_agent("What is the capital of Japan and what's the weather there?")
    # Agent will: get_capital("Japan") → "Tokyo" → get_weather("Tokyo")

    # Test 5: No tool needed
    print("\n--- Q5: No tool needed ---")
    print("User: What is the speed of light?")
    ask_agent("What is the speed of light?")


asyncio.run(main())


# ── WHAT WE LEARNED ──────────────────────────────────────────────────────────
#
# 1. Tools in ADK are just plain Python functions (NO @tool decorator!)
# 2. ADK reads type hints + docstrings to build the tool schema
# 3. Pass tools=[func1, func2] when creating the Agent
# 4. The agent decides WHEN to call a tool (same as LangChain)
# 5. The agent can chain multiple tools in one turn
# 6. The agent can answer without tools when it knows the answer
#
# NEXT: step_03_multi_turn.py — Conversation memory
