import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
import torch
from pathlib import Path
import shutil

def preparar_validacion():
    """Copia etiquetas para el set de validación"""
    print("\n" + "="*60)
    print("PREPARANDO SET DE VALIDACIÓN")
    print("="*60)
    
    train_lbls = Path('data/labels/train')
    val_imgs = Path('data/imagen/val')
    val_lbls = Path('data/labels/val')
    
    # Crear carpeta si no existe
    val_lbls.mkdir(parents=True, exist_ok=True)
    
    # Obtener imágenes de val
    val_images = list(val_imgs.glob('*.*'))
    copied = 0
    
    print(f"Procesando {len(val_images)} imágenes de validación...")
    
    for img in val_images:
        label_name = img.stem + '.txt'
        label_src = train_lbls / label_name
        
        if label_src.exists():
            label_dst = val_lbls / label_name
            shutil.copy(label_src, label_dst)
            copied += 1
    
    print(f"✅ Se copiaron {copied} etiquetas")
    print(f"⚠️  {len(val_images) - copied} imágenes sin etiqueta")
    print("="*60 + "\n")
    
    return copied > 0

def verificar_gpu():
    """Verifica que la GPU esté disponible"""
    print("\n" + "="*60)
    print("VERIFICACIÓN DE GPU")
    print("="*60)
    print(f"PyTorch versión: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print("✅ GPU lista para entrenar")
    else:
        print("⚠️  GPU NO detectada - Se usará CPU (mucho más lento)")
    
    print("="*60 + "\n")
    return torch.cuda.is_available()

def main():
    print("\n" + "🗑️"*30)
    print("ENTRENAMIENTO MEJORADO - DETECTOR DE BASURA")
    print("🗑️"*30 + "\n")
    
    # Verificar GPU
    tiene_gpu = verificar_gpu()
    
    # Preparar validación
    val_ok = preparar_validacion()
    
    if not val_ok:
        print("⚠️  ADVERTENCIA: No se pudieron copiar etiquetas de validación")
        continuar = input("¿Deseas continuar de todas formas? (s/n): ")
        if continuar.lower() != 's':
            print("Entrenamiento cancelado.")
            return
    
    # Seleccionar modelo
    print("\n" + "="*60)
    print("SELECCIÓN DE MODELO")
    print("="*60)
    print("Opciones disponibles:")
    print("  1. yolo11n.pt (Nano)   - Rápido, menos preciso [RECOMENDADO PARA PRUEBAS]")
    print("  2. yolo11s.pt (Small)  - Balance velocidad/precisión")
    print("  3. yolo11m.pt (Medium) - Más preciso, más lento")
    print("  4. yolo11l.pt (Large)  - Muy preciso, requiere más GPU")
    
    modelo_opcion = input("\nSelecciona modelo (1-4) [Por defecto: 2]: ").strip()
    
    modelos = {
        '1': 'yolo11n.pt',
        '2': 'yolo11s.pt',
        '3': 'yolo11m.pt',
        '4': 'yolo11l.pt'
    }
    
    modelo_nombre = modelos.get(modelo_opcion, 'yolo11s.pt')
    print(f"✅ Modelo seleccionado: {modelo_nombre}\n")
    
    # Configuración de entrenamiento
    print("="*60)
    print("CONFIGURACIÓN DE ENTRENAMIENTO")
    print("="*60)
    
    epochs_input = input("Número de epochs [100]: ").strip()
    epochs = int(epochs_input) if epochs_input else 100
    
    batch_input = input("Tamaño de batch [16]: ").strip()
    batch = int(batch_input) if batch_input else 16
    
    imgsz_input = input("Tamaño de imagen (320/416/640) [640]: ").strip()
    imgsz = int(imgsz_input) if imgsz_input else 640
    
    print("\n📋 Configuración:")
    print(f"  - Modelo: {modelo_nombre}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch}")
    print(f"  - Tamaño imagen: {imgsz}")
    print(f"  - Device: {'GPU (CUDA)' if tiene_gpu else 'CPU'}")
    
    confirmar = input("\n¿Iniciar entrenamiento? (s/n): ")
    if confirmar.lower() != 's':
        print("Entrenamiento cancelado.")
        return
    
    # Cargar modelo
    print("\n" + "="*60)
    print("INICIANDO ENTRENAMIENTO")
    print("="*60 + "\n")
    
    model = YOLO(modelo_nombre)
    
    # Entrenar
    results = model.train(
        data='data.yaml',
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0 if tiene_gpu else 'cpu',
        name='basura_detector_mejorado',
        patience=50,
        save=True,
        plots=True,
        workers=0,  # Para evitar problemas en Windows
        
        # Hiperparámetros mejorados
        lr0=0.01,           # Learning rate inicial
        lrf=0.001,          # Learning rate final
        momentum=0.937,     # Momentum
        weight_decay=0.0005,# Weight decay
        warmup_epochs=3.0,  # Epochs de calentamiento
        warmup_momentum=0.8,# Momentum de calentamiento
        
        # Data augmentation
        hsv_h=0.015,        # Variación de matiz
        hsv_s=0.7,          # Variación de saturación
        hsv_v=0.4,          # Variación de brillo
        degrees=10.0,       # Rotación aleatoria
        translate=0.1,      # Traslación
        scale=0.5,          # Escala
        shear=0.0,          # Shearing
        perspective=0.0,    # Perspectiva
        flipud=0.0,         # Flip vertical
        fliplr=0.5,         # Flip horizontal
        mosaic=1.0,         # Mosaic augmentation
        mixup=0.1,          # Mixup augmentation
        copy_paste=0.0,     # Copy-paste augmentation
        
        # Optimización
        optimizer='SGD',    # Optimizador (SGD, Adam, AdamW)
        close_mosaic=10,    # Desactivar mosaic últimos N epochs
        amp=True,           # Automatic Mixed Precision
        
        # Cache para velocidad (usa RAM)
        cache='ram',        # Cachear imágenes en RAM
        
        # Validación
        val=True,           # Validar durante entrenamiento
        
        # Otros
        verbose=True,       # Modo verbose
        seed=0,             # Semilla aleatoria
        deterministic=True, # Modo determinista
        single_cls=False,   # Single class
        rect=False,         # Rectangular training
        cos_lr=False,       # Cosine learning rate
        overlap_mask=True,  # Overlap mask
        mask_ratio=4,       # Mask ratio
        dropout=0.0,        # Dropout
        
        # Guardar
        save_period=-1,     # Guardar cada N epochs (-1 = solo best y last)
    )
    
    print("\n" + "="*60)
    print("✅ ¡ENTRENAMIENTO COMPLETADO!")
    print("="*60)
    print(f"\n📁 Modelo guardado en:")
    print(f"   runs/detect/basura_detector_mejorado/weights/best.pt")
    print(f"\n📊 Resultados y gráficas en:")
    print(f"   runs/detect/basura_detector_mejorado/")
    print("\n💡 Próximos pasos:")
    print("   1. Revisa las gráficas en la carpeta de resultados")
    print("   2. Actualiza app.py para usar el nuevo modelo")
    print("   3. Ejecuta: streamlit run app.py")
    print("\n" + "="*60 + "\n")

if __name__ == '__main__':
    main()