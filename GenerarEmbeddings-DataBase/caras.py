# Importación de librerías necesarias
import os
import numpy as np
import pickle
from PIL import Image
from keras_facenet import FaceNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuración de rutas y nombres de carpetas/archivos
DATA_DIR = 'output_resized_256x256'  # Carpeta con las imágenes originales
PROCESSED_DIR = 'data_procesado'     # Carpeta donde se guardarán imágenes aumentadas
EMBEDDINGS_PATH = 'embeddings.pkl'   # Archivo donde se guardarán los embeddings

# Inicialización del modelo FaceNet
embedder = FaceNet()

def obtener_embedding(img_array):
    """
    Obtiene el embedding facial de una imagen.
    Recibe una imagen como arreglo numpy, la redimensiona a 160x160,
    y usa FaceNet para obtener el vector de características.
    """
    img_resized = np.array(Image.fromarray(img_array).resize((160, 160)))
    embedding = embedder.embeddings([img_resized])
    return embedding[0]

def augment_and_process(input_dir, output_dir):
    """
    Aplica data augmentation a las imágenes en input_dir y guarda los resultados en output_dir.
    Genera 4 imágenes aumentadas por cada imagen original usando rotación, desplazamiento,
    zoom, recorte, y espejado horizontal.
    """
    # Configuración del generador de aumentación
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Crear carpeta de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Recorrer cada persona (carpeta)
    for person in os.listdir(input_dir):
        person_dir = os.path.join(input_dir, person)
        output_person_dir = os.path.join(output_dir, person)
        os.makedirs(output_person_dir, exist_ok=True)

        # Procesar cada imagen de la persona
        for img_file in os.listdir(person_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(person_dir, img_file)
                img = np.array(Image.open(img_path).convert('RGB'))
                
                # Guardar la imagen original
                Image.fromarray(img).save(os.path.join(output_person_dir, f"{os.path.splitext(img_file)[0]}_aug0.jpg"))
                
                # Generar y guardar 4 imágenes aumentadas
                img_batch = img.reshape((1,) + img.shape)
                i = 0
                for batch in datagen.flow(img_batch, batch_size=1):
                    aug_img = batch[0].astype('uint8')
                    Image.fromarray(aug_img).save(os.path.join(output_person_dir, f"{os.path.splitext(img_file)[0]}_aug{i+1}.jpg"))
                    i += 1
                    if i >= 4:
                        break

def generar_embeddings(input_dir):
    """
    Genera los embeddings para todas las imágenes en input_dir.
    Guarda los embeddings en un archivo pickle para uso posterior.
    """
    embeddings = {}
    # Recorrer cada persona (carpeta)
    for person in os.listdir(input_dir):
        embeddings[person] = []
        person_dir = os.path.join(input_dir, person)
        
        # Procesar cada imagen de la persona
        for img_file in os.listdir(person_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(person_dir, img_file)
                img = np.array(Image.open(img_path).convert('RGB'))
                emb = obtener_embedding(img)
                embeddings[person].append(emb)
    
    # Guardar todos los embeddings en un archivo pickle
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(embeddings, f)
    print("Embeddings generados correctamente.")

if __name__ == "__main__":
    # Paso 1: Realizar data augmentation sobre las imágenes originales
    augment_and_process(DATA_DIR, PROCESSED_DIR)
    
    # Paso 2: Generar embeddings a partir de las imágenes aumentadas
    generar_embeddings(PROCESSED_DIR)