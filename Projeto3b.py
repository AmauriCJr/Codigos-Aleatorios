import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,ConfusionMatrixDisplay, confusion_matrix

def evaluate_performance_multiclass(y_test, y_pred):
    
    cm = confusion_matrix(y_true = y_test,y_pred = y_pred)
    
   
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    sensitivity = recall_score(y_test, y_pred, average='macro')
    F1 = f1_score(y_test, y_pred, average='macro')
   
    
    return accuracy, precision,sensitivity, F1, cm
    


if __name__ == '__main__':
    
    #Carrega os dados do mnist e separa o conjunto de treinamento e de teste
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    

    #Deixa os dados no formato de matriz de tons de cinza no formato float32 e normaliza, deixando os valores entre 0 e 1 
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0


    #Aplica one hot encode nas labels
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)


    #Cria a CNN e suas camadas individualmente
    model = Sequential()
    model.add(Conv2D(2, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4, activation='relu'))
    model.add(Dense(10, activation='softmax'))


    #Compila o modelo setando alguns parâmetros
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    #Aplica o modelo nos dadados de treinamento
    model.fit(x_train, y_train, epochs=5, batch_size=128)


    #Realiza predições com os dados de teste
    y_pred = model.predict(x_test)
    
    
    #Salva as predições feitas e as labels de teste em novas variáveis que recebem apenas o valor mais provável
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)


    #Chama a função que calcula as métricas e a matriz de confusão
    accuracy, precision,sensitivity, F1, cm = evaluate_performance_multiclass(y_test_labels, y_pred_labels)
    
    
    
    #Imprime os resultados
    disp = ConfusionMatrixDisplay(confusion_matrix = cm)
    
    fig, ax = plt.subplots(figsize=(10,10))
    
    disp.plot(ax=ax) 
    
    
    
    print("Sensibilidade = %.2f%%"% (sensitivity*100))
    print("Acurácia = %.2f%%"% (accuracy*100))
    print("F1-score = %.2f%%"% (F1*100))
    print("Precisão = %.2f%%"% (precision*100))
    
    
    
    
