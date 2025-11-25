# hello_ai.py - model probe (no unsupported kwargs)
from groq import Groq
import os
import time
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("No GROQ_API_KEY found. Run: set GROQ_API_KEY=your_key_here")

client = Groq(api_key=api_key)

# Candidate models to try (common current/recent Groq names)
candidate_models = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile", 
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

prompt = "Explain Flutter in simple words."

for m in candidate_models:
    print(f"\nTrying model: {m} ...")
    try:
        resp = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": prompt}]
        )
        # If we get here, model accepted request
        print("✅ Success — model works:", m)
        print("AI output preview:\n", resp.choices[0].message.content)
        break
    except Exception as e:
        # Print short error to help diagnose (single-line)
        err = str(e).splitlines()[0]
        print("❌ Error with model", m, "->", err)
        time.sleep(0.2)
else:
    print("\nNo candidate model worked. Next steps:")
    print("1) Open your Groq Console → Models page and copy an exact model id listed there.")
    print("2) Set MODEL_NAME to that exact id and rerun:")
    print("   set MODEL_NAME=the_model_id")
    print("   python hello_ai.py")
