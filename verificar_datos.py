import os
from pathlib import Path

# Rutas
img_train = Path('data/imagen/train')
lbl_train = Path('data/labes/train')

# Contar archivos
imgs = list(img_train.glob('*.*'))
lbls = list(lbl_train.glob('*.txt'))

print(f"Imágenes en train: {len(imgs)}")
print(f"Etiquetas en train: {len(lbls)}")

# Verificar algunos labels
print("\n--- Primeras 3 etiquetas ---")
for i, lbl_file in enumerate(list(lbl_train.glob('*.txt'))[:3]):
    print(f"\nArchivo: {lbl_file.name}")
    with open(lbl_file, 'r') as f:
        content = f.read()
        if content.strip():
            print(content[:200])  # Primeros 200 caracteres
        else:
            print("⚠️ ARCHIVO VACÍO")

# Verificar correspondencia
print("\n--- Verificando correspondencia ---")
sin_label = 0
for img in imgs[:10]:  # Primeras 10 imágenes
    label_file = lbl_train / (img.stem + '.txt')
    if not label_file.exists():
        print(f"❌ Sin label: {img.name}")
        sin_label += 1
        
print(f"\nImágenes sin label (de las primeras 10): {sin_label}")