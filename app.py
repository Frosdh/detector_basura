import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Detector de Basura", layout="wide")

st.title("🗑️ Detector de Basura con YOLOv11")
st.write("Clasifica objetos en: **Orgánicos**, **Inorgánicos** y **Reciclables**")

# Cargar el modelo entrenado
@st.cache_resource
def load_model():
    try:
        # Cambia esta ruta a la carpeta donde se guardó tu modelo
        model = YOLO('runs/detect/basura_detector_mejorado/weights/best.pt')
        return model
    except Exception as e:
        st.error(f"⚠️ Error al cargar el modelo: {e}")
        st.info("Verifica que el modelo esté en: runs/detect/basura_detector_mejorado/weights/best.pt")
        return None

model = load_model()

if model is not None:
    # Sidebar para opciones
    st.sidebar.header("⚙️ Configuración")
    confidence = st.sidebar.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.05)
    
    # Subir imagen
    uploaded_file = st.file_uploader("📤 Sube una imagen para analizar", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Leer imagen
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Imagen Original")
            st.image(image, use_container_width=True)
        
        # Hacer predicción
        with st.spinner('🔍 Analizando imagen...'):
            results = model.predict(image, conf=confidence)
        
        # Mostrar resultados
        with col2:
            st.subheader("🎯 Detección")
            annotated_image = results[0].plot()
            st.image(annotated_image, use_container_width=True)
        
        # Mostrar estadísticas
        st.subheader("📊 Resultados del Análisis")
        
        if len(results[0].boxes) > 0:
            detections = results[0].boxes
            classes = detections.cls.cpu().numpy()
            confidences = detections.conf.cpu().numpy()
            
            # Contar objetos por clase
            class_names = model.names
            counts = {'organicos': 0, 'inorganicos': 0, 'reciclables': 0}
            
            for cls in classes:
                class_name = class_names[int(cls)]
                counts[class_name] += 1
            
            # Mostrar métricas
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🍃 Orgánicos", counts['organicos'])
            with col2:
                st.metric("🗑️ Inorgánicos", counts['inorganicos'])
            with col3:
                st.metric("♻️ Reciclables", counts['reciclables'])
            
            # Tabla de detalles
            st.subheader("📝 Detalle de objetos detectados")
            for i, (cls, conf) in enumerate(zip(classes, confidences)):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"**Objeto {i+1}:** {class_names[int(cls)]}")
                with col_b:
                    st.write(f"Confianza: **{conf:.1%}**")
        else:
            st.warning("⚠️ No se detectaron objetos en la imagen. Intenta con otra imagen o ajusta la confianza mínima.")
    else:
        st.info("👆 Sube una imagen para comenzar el análisis")
        
        # Mostrar información del modelo
        st.sidebar.markdown("---")
        st.sidebar.subheader("ℹ️ Información del Modelo")
        st.sidebar.write(f"**Clases detectadas:** 3")
        st.sidebar.write("- 🍃 Orgánicos")
        st.sidebar.write("- 🗑️ Inorgánicos")  
        st.sidebar.write("- ♻️ Reciclables")
else:
    st.error("No se pudo cargar el modelo. Verifica la instalación.")