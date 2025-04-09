import httpx
from config import GROQ_API_KEY, GROQ_MODEL

def get_therapist_response(user_text: str, sentiment: str):
    prompt = f"You are a friendly and empathetic AI therapist. The user feels {sentiment.lower()}.\nUser: {user_text}\nAI:"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    response = httpx.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
    return response.json()["choices"][0]["message"]["content"]