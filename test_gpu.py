import torch

print("="*60)
print("VERIFICACIÓN DE GPU")
print("="*60)
print(f"PyTorch versión: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Versión CUDA: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"Cantidad de GPUs: {torch.cuda.device_count()}")
    print(f"Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("✅ ¡GPU lista para usar!")
else:
    print("❌ GPU NO detectada")
print("="*60)