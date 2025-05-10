<p align="center">
  <img src="https://mcd.unison.mx/wp-content/themes/awaken/img/logo_mcd.png" alt="MCD" width="120">
</p>

# Reconocimiento Facial para aspirantes aceptados Maestría en Ciencia de Datos (2024)

Este proyecto implementa un sistema de reconocimiento facial en tiempo real que utiliza la cámara para detectar rostros y compararlos contra una base de datos de aspirantes aceptados a la Maestría en Ciencia de Datos.

Usa los modelos **MTCNN** para detección y **FaceNet** para generación de embeddings y comparación.

## Estructura del proyecto
```
├── app.py                      → Aplicación principal Flask
├── templates/                  → Plantillas HTML (frontend)
├── static/                     → Archivos estáticos (logos)
├── GenerarEmbeddings-DataBase/ → Archivos para generar embeddings.pkl
├── embeddings.pkl              → Archivo pickle con embeddings preprocesados
├── requirements.txt            → Lista de dependencias Python
├── Dockerfile                  → Archivo para contenerización con Docker
├── LICENSE                     → Licencia del proyecto
└── README.md                   → Este documento
```

## ¿Qué hace este proyecto?
Este proyecto es la base de una página web interactiva que integra un sistema de reconocimiento facial en tiempo real.

- Captura video en tiempo real usando la cámara web del usuario desde el navegador.
- Detecta rostros automáticamente usando el modelo MTCNN.
- Genera embeddings faciales con FaceNet para cada rostro detectado.
- Compara estos embeddings contra una base de datos preexistente que contiene los rostros de los aspirantes aceptados a la Maestría en Ciencia de Datos.
- Muestra en la interfaz web si el rostro detectado es reconocido o desconocido.
- Permite cambiar entre diferentes cámaras si el dispositivo tiene varias.


## Página Web en Línea
Puedes probar la aplicación directamente en línea:

🌐 [Ir a la página web](https://facereco-production.up.railway.app/)
