import tensorflow as tf
import numpy as np
from PIL import Image

# CARREGAR APENAS UMA VEZ (Fora da função)
# Isso evita ler o disco a cada frame desenhado
MODELO = tf.keras.models.load_model("model/tf_model.h5")

def preverImagem(imagem):
    # Converte o array NumPy (RGBA) para PIL Image, depois para escala de cinza
    img = Image.fromarray(imagem.astype(np.uint8)).convert('L').resize((28, 28))

    img_array = np.array(img, dtype=np.float32) / 255.0

    # INVERSÃO DE CORES CRUCIAL:
    # Se a imagem tiver fundo majoritariamente claro (média > 0.5), invertemos.
    # Isso garante que o fundo seja preto (0) e o desenho seja branco (1)
    if np.mean(img_array) > 0.5:
        img_array = 1.0 - img_array

    img_array = img_array.reshape(1, 28, 28, 1)

    raw_data = MODELO.predict(img_array)

    probs = raw_data[0]
    num_previsto = np.argmax(probs)
    certeza = float(probs[num_previsto]) * 100

    return int(num_previsto), round(certeza, 2)