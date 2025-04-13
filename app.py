from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os

from voice_to_text import transcribe_audio
from sentiment_analysis import analyze_text_sentiment, analyze_voice_features, merge_sentiments
from llm_chat import get_therapist_response
from tts import speak_text

# Initialize FastAPI
app = FastAPI()

# CORS Middleware for allowing frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint for processing voice input
@app.post("/api/voice_input")
async def process_voice(audio: UploadFile):
    try:
        # Read the audio file
        audio_bytes = await audio.read()
        # Transcribe the audio to text
        transcript, wav_bytes = transcribe_audio(audio_bytes)

        # Analyze sentiment from text and voice
        text_sentiment, _ = analyze_text_sentiment(transcript)
        voice_sentiment, _ = analyze_voice_features(wav_bytes)

        # Merge text and voice sentiments
        final_sentiment = merge_sentiments(text_sentiment, voice_sentiment)

        # Get response from AI therapist
        response = get_therapist_response(transcript, final_sentiment)

        # Generate speech for the response
        speak_text(response)

        return {
            "transcript": transcript,
            "text_sentiment": text_sentiment,
            "voice_sentiment": voice_sentiment,
            "final_sentiment": final_sentiment,
            "response": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint for processing text input
@app.post("/api/text_input")
async def process_text(user_text: str = Form(...)):
    try:
        # Analyze text sentiment
        text_sentiment, _ = analyze_text_sentiment(user_text)
        final_sentiment = text_sentiment.lower()

        # Get response from AI therapist
        response = get_therapist_response(user_text, final_sentiment)

        # Generate speech for the response
        speak_text(response)

        return {"sentiment": final_sentiment, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint for serving the generated speech (for download or playback in frontend)
@app.get("/api/get_audio/{filename}")
async def get_audio(filename: str):
    audio_path = os.path.join("audio_files", filename)

    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(audio_path)