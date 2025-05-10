# Importación de librerías necesarias
from flask import Flask, render_template, request, jsonify
import os
import base64
import io
import time
import pickle
import numpy as np
from PIL import Image
import cv2
from threading import Thread
import logging

# Configuración del sistema de logs para mostrar información
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Creación de la aplicación Flask
app = Flask(__name__, static_url_path='/static')

# Variables globales para almacenar los modelos cargados
embedder = None  # Modelo FaceNet para embeddings faciales
detector = None  # Detector MTCNN para rostros
base_datos = None  # Base de datos de embeddings conocidos
models_loaded = False  # Bandera para indicar si los modelos ya están listos
loading_thread = None  # Hilo para la carga asíncrona de modelos

def load_resources():
    """Carga los modelos de ML en un hilo separado"""
    global embedder, detector, base_datos, models_loaded
    
    try:
        # Cargar FaceNet
        logger.info("Cargando FaceNet…")
        t0 = time.time()
        from keras_facenet import FaceNet
        embedder = FaceNet()
        logger.info(f"FaceNet cargado en {time.time() - t0:.2f}s")
        
        # Cargar detector MTCNN
        logger.info("Cargando detector MTCNN…")
        from mtcnn.mtcnn import MTCNN
        detector = MTCNN()
        logger.info("Detector MTCNN cargado")
        
        # Cargar los embeddings previamente guardados
        logger.info("Cargando embeddings…")
        with open("embeddings.pkl", "rb") as f:
            data = pickle.load(f)
            base_datos = {k: [np.array(e) for e in v] for k, v in data.items()}
        logger.info(f"Embeddings cargadas para {len(base_datos)} identidades")
        
        # Marcar como modelos cargados
        models_loaded = True
    except Exception as e:
        logger.error(f"Error cargando modelos: {e}")
        models_loaded = False

# Iniciar la carga de los modelos en segundo plano
loading_thread = Thread(target=load_resources)
loading_thread.daemon = True  # Permite que el hilo termine al cerrar la app
loading_thread.start()

# Ruta principal que carga la página index.html
@app.route("/")
def index():
    return render_template("index.html")

# Ruta para verificar el estado de carga de los modelos
@app.route("/status")
def status():
    global models_loaded, embedder, detector, base_datos
    
    # Verificar si los modelos realmente están cargados (aunque la bandera esté mal)
    actual_models_loaded = (
        embedder is not None and 
        detector is not None and 
        base_datos is not None
    )
    
    # Corregir la bandera si los modelos están cargados
    if actual_models_loaded and not models_loaded:
        models_loaded = True
        logger.info("Models detected as loaded, updating models_loaded flag")
    
    # Preparar la respuesta con el progreso de carga
    progress = {
        "models_loaded": models_loaded,
        "identities": len(base_datos) if models_loaded and base_datos else 0,
        "detector_loaded": detector is not None,
        "embedder_loaded": embedder is not None
    }
    
    logger.info(f"Status check: {progress}")
    return jsonify(progress)

# Ruta POST para reconocer personas en una imagen enviada
@app.route("/reconocer", methods=["POST"])
def reconocer():
    global models_loaded, embedder, detector, base_datos
    
    # Si los modelos aún no están listos, devolver error
    if not models_loaded:
        return jsonify(success=False, error="Modelos aún cargando, intente más tarde"), 503
    
    try:
        logger.info(" /reconocer invocado")
        t_start = time.time()

        # Obtener imagen codificada en base64 desde el JSON enviado
        data = request.get_json(force=True)
        image_data = data.get("image", "")
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]  # Eliminar cabecera del base64
        img_bytes = base64.b64decode(image_data)

        # Convertir bytes a arreglo numpy usando PIL
        img = Image.open(io.BytesIO(img_bytes))
        arr = np.array(img)
        logger.info(f"Imagen recibida: shape={arr.shape}, dtype={arr.dtype}")

        # Detectar rostros usando MTCNN
        detecciones = detector.detect_faces(arr)
        logger.info(f"MTCNN detectó {len(detecciones)} rostros")

        resultados = []
        # Procesar cada rostro detectado
        for cara in detecciones:
            x, y, w, h = cara["box"]
            # Asegurar coordenadas válidas dentro de la imagen
            x, y = max(0, x), max(0, y)
            x2 = min(arr.shape[1], x + w)
            y2 = min(arr.shape[0], y + h)
            face_img = arr[y:y2, x:x2]
            nombre, distancia = reconocer_persona(face_img)
            resultados.append({
                "nombre": nombre,
                "distancia": float(distancia),
                "bbox": [int(x), int(y), int(w), int(h)]
            })

        logger.info(f"Procesado en {time.time() - t_start:.2f}s")
        return jsonify(success=True, resultados=resultados)

    except Exception as e:
        logger.error(f"Error en /reconocer: {e}")
        return jsonify(success=False, error=str(e)), 500

def reconocer_persona(img_array, umbral=0.8):
    """Compara la imagen de rostro detectada contra la base de datos"""
    from numpy.linalg import norm

    try:
        # Garantizar que la imagen esté en formato RGB
        if img_array.ndim == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif img_array.shape[2] != 3:
            raise ValueError(f"Imagen con forma inesperada: {img_array.shape}")

        # Redimensionar rostro a 160x160 píxeles
        face_resized = cv2.resize(img_array, (160, 160))

        # Obtener embedding usando FaceNet
        embedding = embedder.embeddings([face_resized])[0]

        # Comparar embedding contra cada identidad en la base de datos
        nombre_identificado = "Desconocido"
        distancia_minima = float("inf")
        for nombre, emb_list in base_datos.items():
            for emb_base in emb_list:
                d = norm(embedding - emb_base)
                if d < distancia_minima and d < umbral:
                    distancia_minima = d
                    nombre_identificado = nombre
        return nombre_identificado, distancia_minima

    except Exception as e:
        logger.error(f"Error procesando imagen para reconocimiento: {e}")
        return "Error", float("inf")

# Punto de entrada principal para arrancar la app Flask
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Arrancando servidor en 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
