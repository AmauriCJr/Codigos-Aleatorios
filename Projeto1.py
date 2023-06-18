import os
import numpy as np
import pandas as pd
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



def read_names(path):
    x, y, file_names_array = [], [], []


    #Gera o endereço de todos os arquivos .csv na pasta
    file_paths = glob.glob(path + '/*.csv') 

    #Abre arquivo por arquivo, soma todas as colunas e salva em um array x,
    #a label que informa a presença ou não da patologia está salva no nome do arquivo, no primeiro caractér,
    #essa é salva em um array y
    for i in file_paths:
        df = pd.read_csv(i)
    
        row_sum = (np.sum(df.values, axis=1))/12
        
    
        x.append(row_sum)
    
        file_name = os.path.splitext(os.path.basename(i))[0]
    
        file_names_array.append(file_name)
        
        first_letter = file_name[0]
    
        y.append(first_letter)
        
    return np.array(x), np.array(y), file_name


def evaluate_performance(y_test, y_pred):
    
    #Calcula as métricas do modelo a partir da matriz de confusão.
    cm = confusion_matrix(y_test, y_pred)
    
    tp = cm[1,1]
    tn = cm[0,0]
    fn = cm[1,0]
    fp = cm[0,1]
    p = tp + fn
    n = tn + fp
    accuracy = (tp + tn) / (p + n)  
    precision = tp / (tp + fp)      
    sensitivity = tp/p              
    specificity = tn/n
    F1 = (2*tp)/(2*tp + fp + fn)
    return tp, tn, fp, fn, accuracy, precision,sensitivity,specificity, F1, cm


if __name__ == '__main__':
    
    name_path = r'C:\Users\amaur\OneDrive\Área de Trabalho\Eletrônica\Mestrado\Aulas\Conjunto de Dados Projeto 1\Dados 2'
    
    x, y, id_paciente = read_names(name_path)
    
    
    
    
    
    #Aplica one hot encode na variável y
    y = tf.keras.utils.to_categorical(y)
    
    
    #Separa o conjunto de dados em dados de treinamento e dados de teste
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    

    #Cria o modelo MLP definindo três camadas e suas características
    model = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu', input_shape=(5000,)),
                                  tf.keras.layers.Dense(64, activation='relu'),
                                  tf.keras.layers.Dense(2, activation='softmax')])


    #Compila o modelo criado aplicando o otimizador e a função loss
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    #Treina o modelo com os dados de treinamento
    model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test))

    #Aplica o modelo ao conjunto de dados de teste
    y_pred = model.predict(x_test)
    
    #Desfaz o one hot encode pegando a posição do valor máximo dos pares 0 = (1 , 0) e 1 = (0 , 1)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    
    
    #Calcula as métricas do modelo
    tp, tn, fp, fn, accuracy, precision, sensitivity, specificity, F1, cm = evaluate_performance(y_test, y_pred)
    
    
    print("sensitivity = %.2f%%" % (sensitivity*100))
    print("specificity = %.2f%%" % (specificity*100))
    print("accuracy = %.2f%%" % (accuracy*100))
    print("F1 = %.2f%%" % (F1*100))
    print("precision = %.2f%%" % (precision*100))
    
    print("Confusion Matrix: ")
    print(cm)
    
    
    cm = confusion_matrix(y_test, y_pred)

    class_labels = ['Classe 0', 'Classe 1']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Matriz de Confusão')
    plt.xlabel('Classe Prevista')
    plt.ylabel('Classe Real')
    plt.show()
