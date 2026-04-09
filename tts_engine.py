import os
import base64
import tempfile
import logging
import json
import requests
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

class GoogleTTS:
    def __init__(self):
        self.whisper = None
        # Intenta tomar la clave de las variables de entorno, o usa la que proporcionaste como fallback
        self.api_key = os.environ.get('GOOGLE_TTS_API_KEY', 'AIzaSyCEUsJaFsZgTXp63w350LeaR3MVgqVmejw')

    def _ensure_loaded(self):
        if self.whisper is None:
            logger.info("Cargando Faster-Whisper (tiny, CPU) para sincronizacion milimetrica de subtitulos...")
            self.whisper = WhisperModel("tiny", device="cpu", compute_type="int8")

    def generate(self, text, voice="es-US-Neural2-B", speed=1.0, lang="es-US"):
        self._ensure_loaded()
        
        logger.info(f"Generando audio Premium con Google TTS API | Modelo: {voice} | Velocidad: {speed}")
        
        url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={self.api_key}"
        
        payload = {
            "input": {"text": text},
            "voice": {"languageCode": lang, "name": voice},
            "audioConfig": {
                "audioEncoding": "MP3",
                "speakingRate": speed
            }
        }
        
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code != 200:
            logger.error(f"Error Google TTS: {response.text}")
            raise Exception(f"Error de Google TTS API: {response.status_code}")
            
        audio_b64 = response.json().get("audioContent", "")
        if not audio_b64:
            raise Exception("Google TTS devolvió 200 pero no hay audioContent")
            
        # Decodificamos el MP3 devuelto por la nube
        audio_bytes = base64.b64decode(audio_b64)
        
        with tempfile.TemporaryDirectory() as tmp:
            mp3_path = os.path.join(tmp, "speech.mp3")
            with open(mp3_path, "wb") as f:
                f.write(audio_bytes)
            
            # Pasar el audio puro por Whisper local para obtener timestamps de cada palabra
            segments, _ = self.whisper.transcribe(mp3_path, word_timestamps=True, initial_prompt=text)
            
            word_timestamps = []
            for segment in segments:
                for word in segment.words:
                    word_timestamps.append({
                        "word": word.word.strip(),
                        "start": round(word.start, 3),
                        "end": round(word.end, 3)
                    })
            
            return audio_b64, word_timestamps

# Instancia global para ser usada en app.py
_engine = None

def generate_tts_local(text, voice="es-US-Neural2-B", rate="-5%"):
    global _engine
    if _engine is None:
        _engine = GoogleTTS()
    
    # Fuerzo la voz Neural2 de google si en app.py quedó guardada la antigua
    if "neural2" not in voice.lower():
        voice = "es-US-Neural2-B"

    # Convertir rate (ej: "-5%") a multiplicador de velocidad (ej: 0.95)
    speed = 1.0
    try:
        if isinstance(rate, str) and rate.endswith('%'):
            speed = 1.0 + float(rate[:-1]) / 100.0
    except:
        pass
        
    return _engine.generate(text, voice=voice, speed=speed, lang="es-US")
