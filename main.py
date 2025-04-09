from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

from voice_to_text import transcribe_audio
from sentiment_analysis import analyze_text_sentiment
from llm_chat import get_therapist_response
from tts import speak_text

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/voice_input")
async def process_voice(audio: UploadFile):
    audio_bytes = await audio.read()
    text = transcribe_audio(audio_bytes)
    sentiment, confidence = analyze_text_sentiment(text)
    response = get_therapist_response(text, sentiment)
    speak_text(response)
    return {"transcript": text, "sentiment": sentiment, "response": response}


@app.post("/api/text_input")
async def process_text(user_text: str = Form(...)):
    sentiment, confidence = analyze_text_sentiment(user_text)
    response = get_therapist_response(user_text, sentiment)
    speak_text(response)
    return {"sentiment": sentiment, "response": response}