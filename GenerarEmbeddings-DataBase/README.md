# Generador de Embeddings Faciales con FaceNet

Este proyecto realiza dos tareas principales:
1. Aplica **data augmentation** sobre un conjunto de imágenes organizadas por carpetas (una carpeta por persona).
2. Genera **embeddings faciales** usando FaceNet para todas las imágenes aumentadas y guarda los resultados en un archivo `.pkl`.

---

## Estructura esperada de carpetas
```
├── output_resized_256x256/         → Base de datos
├── caras.py                        → Aplicación para generar el Embeddings
├── requirements-embeddings.txt     → Lista de dependencias Python
└── README.md                       → Este documento
```


Después del procesamiento, se generará:

```
├── output_resized_256x256/         → Base de datos
├── data_procesado/                 → Imagenes (data augmentation) para generar el Embeddings
├── caras.py                        → Aplicación para generar el Embeddings
├── embeddings.pkl                  → Archivo pickle con embeddings preprocesados
├── requirements-embeddings.txt     → Lista de dependencias Python
└── README.md                       → Este documento
```
s

## Requisitos

Instala las dependencias usando:
```
pip install -r requirements.txt
```

## Cómo correr el script

1. Asegúrate de tener las imágenes originales en la carpeta: output_resized_256x256/

2. Corre el script principal:
```
python script.py
```
Esto ejecutará:
- Aplicar aumentación de datos: se guarda en data_procesado/
- Generar embeddings: se guarda en embeddings.pkl

## Resultado final
Carpeta data_procesado/ con imágenes originales + aumentadas.

Archivo embeddings.pkl que contiene un diccionario:
```
{
    'persona1': [embedding1, embedding2, ...],
    'persona2': [embedding1, embedding2, ...],
    ...
}
```
Cada embedding es un vector NumPy generado por FaceNet.

## Notas
- Asegúrate de usar imágenes RGB (no en blanco y negro).
- Si deseas cambiar la cantidad de imágenes aumentadas, modifica el parámetro i >= 4 en la función augment_and_process.
