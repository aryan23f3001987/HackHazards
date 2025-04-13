# ---------- app.py ----------
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

from voice_to_text import transcribe_audio
from sentiment_analysis import (
    analyze_text_sentiment,
    analyze_voice_features,
    merge_sentiments
)
from llm_chat import get_therapist_response
from tts import speak_text

app = FastAPI()

# Enable CORS so frontend apps can access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/voice_input")
async def process_voice(audio: UploadFile):
    # Read and convert audio input
    audio_bytes = await audio.read()
    transcript, wav_bytes = transcribe_audio(audio_bytes)

    # Analyze sentiment from both text and voice
    text_sentiment, _ = analyze_text_sentiment(transcript)
    voice_sentiment, _ = analyze_voice_features(wav_bytes)

    # Merge both to get final sentiment
    final_sentiment = merge_sentiments(text_sentiment, voice_sentiment)

    # Get AI therapist response
    response = get_therapist_response(transcript, final_sentiment)

    # Speak the response out loud
    speak_text(response)

    return {
        "transcript": transcript,
        "text_sentiment": text_sentiment,
        "voice_sentiment": voice_sentiment,
        "final_sentiment": final_sentiment,
        "response": response
    }

@app.post("/api/text_input")
async def process_text(user_text: str = Form(...)):
    # Text-only sentiment analysis
    text_sentiment, _ = analyze_text_sentiment(user_text)
    final_sentiment = text_sentiment.lower()

    # LLM therapist reply
    response = get_therapist_response(user_text, final_sentiment)

    # Speak out the reply
    speak_text(response)

    return {
        "sentiment": final_sentiment,
        "response": response
    }