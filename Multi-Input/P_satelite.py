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

# Diretório para os arquivos de label
labels_dir = "C:/IFPR-CONTEUDO/GITHUB/stationdatain-streamlit/Multi-Input/preprocessed_images/labels"

# Número de imagens a serem selecionadas de cada tipo
num_images = 21

# Cria a pasta de saída, se não existir
os.makedirs(output_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Seleciona as melhores imagens JPG
best_jpg_images = select_best_images(input_dir, num_images, "jpg")

# Pré-processa e salva as imagens e cria os arquivos de label
for i, filename in enumerate(best_jpg_images):
    image_path = os.path.join(input_dir, filename)

    # Cria o arquivo de label
    label_filename = f"jpg_{i}_preprocessed.txt"
    label_path = os.path.join(labels_dir, label_filename)
    with open(label_path, "w") as f:
        f.write("1")  # Escreve o label 1 para as imagens JPG

    # Pré-processamento
    preprocessed_image = preprocess_image(image_path, resize_shape=(224, 224))
    if preprocessed_image is not None:
        output_filename = f"jpg_{i}_preprocessed.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, preprocessed_image)

print("Imagens pré-processadas e arquivos de label criados com sucesso!")