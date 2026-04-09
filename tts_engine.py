import numpy as np
import os
import base64
import subprocess
import tempfile
import logging
import json
import requests
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

    def _download_file(self, url, dest):
        logger.info(f"Descargando archivo necesario de {url}...")
        try:
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            with open(dest, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0 and downloaded % (1024 * 1024 * 10) == 0: # Log cada 10MB
                            logger.info(f"Progreso: {downloaded / (1024 * 1024):.1f}MB / {total_size / (1024 * 1024):.1f}MB")
            logger.info(f"Descarga completada: {dest}")
        except Exception as e:
            logger.error(f"Error descargando {url}: {e}")
            if os.path.exists(dest):
                os.remove(dest)
            raise

    def _ensure_loaded(self):
        # Verificar y descargar modelos si no existen o están incompletos
        if not os.path.exists(self.model_path) or os.path.getsize(self.model_path) < 1024 * 1024:
            # Usamos el modelo estable de thewh1teagle
            url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
            self._download_file(url, self.model_path)
            
        if not os.path.exists(self.voices_path) or os.path.getsize(self.voices_path) < 1024 * 1024:
            # Usamos el paquete de voces multilingüe de rany2 (incluye Alex, Dora, etc.)
            url = "https://github.com/rany2/kokoro-onnx/releases/download/v0.1.0/voices.bin"
            self._download_file(url, self.voices_path)

        if self.model is None:
            logger.info("Cargando Kokoro TTS ONNX...")
            self.model = Kokoro(self.model_path, self.voices_path)
            # Log de voces para confirmar que Alex/Dora están presentes
            if hasattr(self.model, 'voices'):
                logger.info(f"Cargadas {len(self.model.voices)} voces del catálogo.")
            
        if self.whisper is None:
            logger.info("Cargando Faster-Whisper (tiny, CPU)...")
            # compute_type="int8" para máxima eficiencia en CPU
            self.whisper = WhisperModel("tiny", device="cpu", compute_type="int8")

    def generate(self, text, voice="em_alex", speed=1.0, lang="es-419"):
        self._ensure_loaded()
        
        # Mapeo robusto a voces de Kokoro v1.0
        # Kokoro usa prefijos: es_ (Spanish Spain), em_ (Male), ef_ (Female), am_ (American Male), etc.
        voice_lower = voice.lower()
        if "es" in lang.lower() or voice_lower.startswith("es-"):
            lang = "es-419" # Dialecto latinoamericano
            # Mapeo de nombres genéricos a la voz de Alex (la mejor masculina latina)
            if not (voice_lower.startswith('es_') or voice_lower.startswith('em_') or voice_lower.startswith('ef_')):
                voice = "em_alex"
        
        # Verificación final: si la voz no está en el catálogo, usamos una por defecto segura
        if self.model and hasattr(self.model, 'voices'):
            available_voices = list(self.model.voices.keys())
            if voice not in available_voices:
                logger.warning(f"Voz '{voice}' no encontrada en catálogo Kokoro. Disponibles: {available_voices}")
                # Si buscamos español, intentamos es_fede primero, luego cualquier es_
                if "es" in lang:
                    voice = "es_fede" if "es_fede" in available_voices else next((v for v in available_voices if v.startswith('es_')), available_voices[0])
                else:
                    voice = "af_heart" if "af_heart" in available_voices else available_voices[0]
                logger.info(f"Reasignada voz a: {voice}")
            
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

def generate_tts_local(text, voice="em_alex", rate="-5%"):
    global _engine
    if _engine is None:
        # Priorizamos el directorio persistente para los modelos
        persistent_dir = "/root/shorts_data"
        if not os.path.exists(persistent_dir):
             # Fallback al directorio actual si no estamos en Docker/EasyPanel
             persistent_dir = os.path.dirname(os.path.abspath(__file__))
             
        m_path = os.path.join(persistent_dir, "kokoro-v1.0.onnx")
        v_path = os.path.join(persistent_dir, "voices-v1.0.bin")
        _engine = KokoroTTS(m_path, v_path)
    
    # Convertir rate (ej: "-5%") a multiplicador de velocidad (ej: 0.95)
    speed = 1.0
    try:
        if rate.endswith('%'):
            speed = 1.0 + float(rate[:-1]) / 100.0
    except:
        pass
        
    return _engine.generate(text, voice=voice, speed=speed, lang="es-419")
