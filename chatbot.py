# chatbot.py
from groq import Groq
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# --- config ---
API_KEY = os.environ.get("GROQ_API_KEY")
MODEL = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
# optional system prompt to control assistant behaviour
SYSTEM_PROMPT = (
    "You are a helpful assistant. Keep answers concise and clear. "
    "If asked to write code, include short explanations."
)

if not API_KEY:
    raise ValueError("No GROQ_API_KEY found. Run: set GROQ_API_KEY=your_key_here")

client = Groq(api_key=API_KEY)

print("🤖 Chatbot ready. Type 'exit' or Ctrl+C to quit.\n")

# messages list holds the conversation history in chat format
messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

try:
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("🤖 Bot: Goodbye!")
            break

        # append user message
        messages.append({"role": "user", "content": user_input})

        # call the model
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages
            )
            bot_reply = resp.choices[0].message.content
            print("\n🤖 Bot:", bot_reply, "\n")

            # append assistant reply so the next call has history
            messages.append({"role": "assistant", "content": bot_reply})

            # optional: keep history short to avoid long inputs (trim oldest user msgs)
            # if len(messages) > 20:
            #     # keep system prompt + last 18 messages
            #     messages = [messages[0]] + messages[-18:]

        except Exception as e:
            print("❌ Error calling model:", str(e).splitlines()[0])
            # If error, remove last user message so we don't resend it repeatedly
            if messages and messages[-1]["role"] == "user":
                messages.pop()

except KeyboardInterrupt:
    print("\n\nExiting — bye!")
    sys.exit(0)
