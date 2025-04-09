import whisper
from pydub import AudioSegment
from io import BytesIO

model = whisper.load_model("base")

def transcribe_audio(audio_bytes: bytes) -> str:
    audio = AudioSegment.from_file(BytesIO(audio_bytes), format="webm")
    wav_io = BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    result = model.transcribe(wav_io)
    return result["text"]