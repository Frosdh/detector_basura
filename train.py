import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
import torch

def main():
    # Verificar GPU
    print("\n" + "="*50)
    print("VERIFICACIÓN DE GPU")
    print("="*50)
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("="*50 + "\n")

    # Cargar modelo
    model = YOLO('yolo11n.pt')

    # Entrenar usando GPU
    print("Iniciando entrenamiento...")
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        name='basura_detector',
        patience=50,
        save=True,
        plots=True,
        workers=0,  # Cambiado a 0 para evitar problemas en Windows
        cache=True
    )

    print("\n✅ ¡Entrenamiento completado!")
    print(f"📁 Modelo guardado en: runs/detect/basura_detector/weights/best.pt")

if __name__ == '__main__':
    main()