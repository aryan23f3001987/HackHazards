import pyttsx3

tts_engine = pyttsx3.init()

def speak_text(text: str):
    tts_engine.say(text)
    tts_engine.runAndWait()