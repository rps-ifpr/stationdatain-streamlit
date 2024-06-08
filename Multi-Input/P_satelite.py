import os
import cv2
import numpy as np
from PIL import Image
import random

def preprocess_image(image_path, resize_shape=(224, 224), grayscale=False):
    """
    Pré-processa uma imagem.
    """
    try:
        if image_path.lower().endswith('.tiff'):
            image = Image.open(image_path)
            image = np.array(image)  # Converte para NumPy array
        elif image_path.lower().endswith('.jpg'):
            image = cv2.imread(image_path)
        else:
            raise ValueError("Formato de imagem inválido. Use TIFF ou JPG.")

        if grayscale:
            if image_path.lower().endswith('.tiff'):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, resize_shape)
        return image

    except Exception as e:
        print(f"Erro ao pré-processar a imagem {image_path}: {e}")
        return None

def select_best_images(input_dir, num_images, file_type):
    """
    Seleciona as melhores imagens de um tipo específico.
    """
    image_paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(file_type.lower())
    ]
    selected_images = random.sample(image_paths, num_images)
    return selected_images

# Diretório de entrada com as imagens
input_dir = "C:/IFPR-CONTEUDO/GITHUB/stationdatain-streamlit/Multi-Input/img_satelite"

# Diretório de saída para as imagens pré-processadas
output_dir = "C:/IFPR-CONTEUDO/GITHUB/stationdatain-streamlit/Multi-Input/preprocessed_images"

# Número de imagens a serem selecionadas de cada tipo
num_tiff_images = 21  # Ajuste para o número de imagens TIFF
num_jpg_images = 18  # Ajuste para o número de imagens JPG

# Seleciona as melhores imagens TIFF
best_tiff_images = select_best_images(input_dir, num_tiff_images, "tiff")

# Seleciona as melhores imagens JPG
best_jpg_images = select_best_images(input_dir, num_jpg_images, "jpg")

# Cria a pasta de saída, se não existir
os.makedirs(output_dir, exist_ok=True)

# Pré-processa e salva as imagens
for i, filename in enumerate(best_tiff_images):
    image_path = os.path.join(input_dir, filename)
    preprocessed_image = preprocess_image(image_path, resize_shape=(224, 224))
    if preprocessed_image is not None:
        output_filename = f"tiff_{i}_preprocessed.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, preprocessed_image)

for i, filename in enumerate(best_jpg_images):
    image_path = os.path.join(input_dir, filename)
    preprocessed_image = preprocess_image(image_path, resize_shape=(224, 224))
    if preprocessed_image is not None:
        output_filename = f"jpg_{i}_preprocessed.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, preprocessed_image)

print("Imagens pré-processadas salvas com sucesso!")