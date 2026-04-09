FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgomp1 \
    espeak-ng \
    fonts-dejavu-core \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Descargar los modelos y voces nativas de Kokoro (Paquete Multilingüe Completo)
RUN wget -q https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx -O kokoro-v1.0.onnx && \
    wget -q https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices.bin -O voices-v1.0.bin

# Copiar dependencias primero para aprovechar el cache de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copiar el codigo del servicio
COPY app.py .
COPY tts_engine.py .
COPY setup_kokoro.py .

# Crear directorio para persistencia si no existe (aunque se mapee por volumen)
RUN mkdir -p /root/shorts_data

EXPOSE 8000

# Iniciamos el servidor
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
