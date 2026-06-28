# ============================================================
# GOOGLE ADK — STEP 07: ADK Web UI + CLI Commands
# Goal: Test agents using ADK's built-in web interface
# Run:  adk web web_agent/
# ============================================================

# ── ADK'S BUILT-IN WEB UI ─────────────────────────────────────────────────────
#
# ADK comes with a chat web interface for testing!
# LangChain has NOTHING like this built-in.
#
# To use it:
#   1. Create a folder with __init__.py + agent.py
#   2. Run: adk web <folder>/
#   3. Open browser at http://localhost:8000
#   4. Chat with your agent interactively!
#
# ── FOLDER STRUCTURE REQUIRED ─────────────────────────────────────────────────
#
# web_agent/
# ├── __init__.py     ← Must export 'agent' variable
# └── agent.py        ← Defines your agent
#
# ── HOW TO RUN ────────────────────────────────────────────────────────────────
#
# cd c:\Code\Google_ADK
# adk web web_agent/
#
# This opens a browser-based chat interface where you can:
# - Send messages to your agent
# - See tool calls in real-time
# - View session history
# - Test multi-turn conversations


# ── ALL ADK CLI COMMANDS ──────────────────────────────────────────────────────
#
# adk web <folder>     → Web UI for interactive testing
# adk run <folder>     → Terminal-based interactive chat
# adk eval <folder>    → Run automated evaluation tests
# adk deploy <folder>  → Deploy to Google Cloud / Vertex AI
#
# Compare to LangChain:
#   LangChain has no built-in CLI like this.
#   You'd need: langserve + custom frontend + manual deploy scripts.


# ── SETUP THE WEB AGENT FOLDER ────────────────────────────────────────────────
#
# The web_agent/ folder is created alongside this file.
# See web_agent/__init__.py and web_agent/agent.py
#
# TO TEST:
#   cd c:\Code\Google_ADK
#   adk web web_agent/
#
# Try these in the web UI:
#   "What's the weather in Tokyo?"
#   "Calculate 125 * 37"
#   "What time is it in London?"
#   "What is an API?"

print("""
============================================================
GOOGLE ADK — STEP 07: Web UI & CLI
============================================================

To launch the web UI, run this in your terminal:

  cd c:\\Code\\Google_ADK
  adk web web_agent/

Then open http://localhost:8000 in your browser.

For terminal-based chat:
  adk run web_agent/

============================================================
""")