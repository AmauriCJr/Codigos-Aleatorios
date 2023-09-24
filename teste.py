import tensorflow as tf

# Verifica se a GPU está disponível e configurada corretamente
if tf.test.gpu_device_name():
    print('GPU encontrada:')
    print(tf.test.gpu_device_name())
else:
    print('GPU não encontrada. Certifique-se de que o CUDA está configurado corretamente.')

# Carrega os dados MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normaliza as imagens para o intervalo [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Cria um modelo simples
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Camada Flatten para redimensionar as imagens
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compila o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treina o modelo
model.fit(x_train, y_train, epochs=5)

# Avalia o modelo
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nAcurácia do teste:', test_acc)