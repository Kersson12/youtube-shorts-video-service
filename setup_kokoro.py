import os
import requests
import sys

def download_file(url, filename):
    print(f"Descargando {filename} desde {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 # 1MB
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            f.write(data)
    print(f"Descarga de {filename} completada.")

def setup():
    # Modelos de Kokoro v1.0
    models = {
        "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
    }
    
    for filename, url in models.items():
        if not os.path.exists(filename):
            download_file(url, filename)
        else:
            print(f"El archivo {filename} ya existe, saltando descarga.")
            
    # Crear carpeta de datos persistentes
    # Usamos ruta expandida para AWS (~/shorts_data)
    # Pero para Windows local también funcionará en el home del usuario
    data_dir = os.path.expanduser("~/shorts_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Directorio creado: {data_dir}")
    else:
        print(f"Directorio ya existe: {data_dir}")

    print("\n--- Setup de Kokoro TTS completado con éxito ---")
    print("RECUERDA instalar las dependencias con:")
    print("pip install kokoro-onnx onnxruntime faster-whisper soundfile requests")

if __name__ == "__main__":
    setup()
