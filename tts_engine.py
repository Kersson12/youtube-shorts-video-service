import os
import base64
import subprocess
import tempfile
import logging
import json
import soundfile as sf
from kokoro_onnx import Kokoro
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

class KokoroTTS:
    def __init__(self, model_path="kokoro-v1.0.onnx", voices_path="voices-v1.0.bin"):
        self.model_path = model_path
        self.voices_path = voices_path
        self.model = None
        self.whisper = None
        
    def _ensure_loaded(self):
        if self.model is None:
            if not os.path.exists(self.model_path) or not os.path.exists(self.voices_path):
                raise Exception(f"Modelos de Kokoro no encontrados en {self.model_path} o {self.voices_path}. Ejecuta el setup primero.")
            logger.info("Cargando Kokoro TTS ONNX...")
            self.model = Kokoro(self.model_path, self.voices_path)
            
        if self.whisper is None:
            logger.info("Cargando Faster-Whisper (tiny, CPU)...")
            # compute_type="int8" para máxima eficiencia en CPU
            self.whisper = WhisperModel("tiny", device="cpu", compute_type="int8")

    def generate(self, text, voice="ef_dora", speed=1.0, lang="es-es"):
        self._ensure_loaded()
        
        # Mapeo robusto a voces de Kokoro v1.0
        # Kokoro usa prefijos: ef_ (Spanish Female), em_ (Spanish Male), af_ (US Female), etc.
        voice_lower = voice.lower()
        if "es" in lang.lower() or "es-" in voice_lower:
            lang = "es-es"
            # Si recibimos una voz de Google/ElevenLabs o af_heart, mapeamos a una española de Kokoro
            if not voice_lower.startswith('e'):
                # Por defecto usamos ef_dora (femenina española)
                voice = "ef_dora"
        
        # Verificación final: si la voz no está en el catálogo, usamos una por defecto segura
        # model.voices es un diccionario con los nombres de las voces cargadas
        available_voices = list(self.model.voices.keys())
        if voice not in available_voices:
            logger.warning(f"Voz '{voice}' no encontrada. Usando 'ef_dora' por defecto. Disponibles: {available_voices}")
            voice = "ef_dora" if "es" in lang else "af_heart"
        
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = os.path.join(tmp, "speech.wav")
            mp3_path = os.path.join(tmp, "speech.mp3")
            
            # 1. Generar Audio con Kokoro
            samples, sample_rate = self.model.create(text, voice=voice, speed=speed, lang=lang)
            sf.write(wav_path, samples, sample_rate)
            
            # 2. Obtener Timestamps con Faster-Whisper
            # Usamos el texto original como 'initial_prompt' para guiar la precisión
            segments, _ = self.whisper.transcribe(wav_path, word_timestamps=True, initial_prompt=text)
            
            word_timestamps = []
            for segment in segments:
                for word in segment.words:
                    word_timestamps.append({
                        "word": word.word.strip(),
                        "start": round(word.start, 3),
                        "end": round(word.end, 3)
                    })
            
            # 3. Convertir a MP3 usando FFmpeg
            subprocess.run([
                'ffmpeg', '-y', '-i', wav_path, 
                '-codec:a', 'libmp3lame', '-qscale:a', '2', 
                mp3_path
            ], capture_output=True, check=True)
            
            with open(mp3_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode('utf-8')
                
            return audio_b64, word_timestamps

# Instancia global para ser usada en app.py
_engine = None

def generate_tts_local(text, voice="ef_dora", rate="-5%"):
    global _engine
    if _engine is None:
        # Buscamos los modelos en la carpeta del script o en shorts_data
        base_dir = os.path.dirname(os.path.abspath(__file__))
        m_path = os.path.join(base_dir, "kokoro-v1.0.onnx")
        v_path = os.path.join(base_dir, "voices-v1.0.bin")
        _engine = KokoroTTS(m_path, v_path)
    
    # Convertir rate (ej: "-5%") a multiplicador de velocidad (ej: 0.95)
    speed = 1.0
    try:
        if rate.endswith('%'):
            speed = 1.0 + float(rate[:-1]) / 100.0
    except:
        pass
        
    return _engine.generate(text, voice=voice, speed=speed, lang="es-es")
