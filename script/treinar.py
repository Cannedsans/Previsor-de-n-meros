import os
import tensorflow as tf
from tensorflow.keras import layers, models

# Scritp gerado por IA pois não achei modelo descente 
print("📥 Baixando MNIST e criando o seu modelo...")

# 1. Carregar dados do MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# 2. Criar a arquitetura CNN clássica
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') # Softmax garante a porcentagem de certeza direta
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. Treinar super rápido (apenas 2 épocas já batem ~98% de acurácia)
model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))

# 4. Criar a pasta model se não existir e salvar
os.makedirs("model", exist_ok=True)
model.save("model/tf_model.h5")

print("\n✅ O modelo 'model/tf_model.h5' foi gerado com sucesso e está pronto para uso!")