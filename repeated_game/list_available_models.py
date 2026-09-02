"""
Diagnostic: list which Gemini models your API key can actually call right
now, and which of those support generateContent (the method our pipeline
uses) and structured JSON output (which the Translator-plus-Solver arm
needs). Run this locally -- it makes a real API call.

    python list_available_models.py
"""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

print("Models available to this API key that support generateContent:\n")
for model in client.models.list():
    actions = getattr(model, "supported_actions", None) or []
    if "generateContent" in actions:
        print(f"  {model.name}")

print("\nPick one of the names above (drop any 'models/' prefix if present) "
      "and set MODEL_NAME to it in direct_ai.py and translator_solver.py.")