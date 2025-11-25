from groq import Groq
import os

from dotenv import load_dotenv

load_dotenv()

# Load API key
API_KEY = os.environ.get("GROQ_API_KEY")
MODEL = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")

if not API_KEY:
    raise ValueError("No GROQ_API_KEY found. Run: set GROQ_API_KEY=your_key_here")

client = Groq(api_key=API_KEY)

def summarize(text, style="short"):
    """
    Summarize the input text.
    style options: short, medium, long, bullet
    """

    # choose summary instructions
    if style == "short":
        instruction = "Summarize the text in 3–4 lines using simple words."
    elif style == "medium":
        instruction = "Give a clear paragraph summary of the following text."
    elif style == "long":
        instruction = "Give a detailed summary with key points."
    elif style == "bullet":
        instruction = "Summarize the text into bullet points."
    else:
        instruction = "Summarize the following text."

    prompt = f"{instruction}\n\nTEXT:\n{text}"

    # call LLM
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    print("📝 Text Summarizer")
    print("Paste your text below. Write '/done' on a new line to finish.\n")

    lines = []
    while True:
        line = input()
        if line.strip().lower() == "/done":
            break
        lines.append(line)

    full_text = "\n".join(lines)
    print("\nChoose summary type: short / medium / long / bullet")
    style = input("Type: ").strip().lower()

    print("\n🔍 Generating summary...\n")
    result = summarize(full_text, style)
    print("📌 Summary:\n")
    print(result)
