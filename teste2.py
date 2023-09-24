import os
import numpy as np
import pandas as pd
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import savemat
import seaborn



def read_data(folder_path):
    file_paths = glob.glob(os.path.join(folder_path, '*.txt'))  

    data_list = []  

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            content = file.read()  
            data_list.append(content)  

    return data_list

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



def split_to_float(data_list):
    
    data_list = [value.splitlines() for value in data_list]
    data_list = [[float(line) for line in value] for value in data_list]
    
    return data_list

def frequency_bands_energy(x, window_duration_seconds, window_type, fs_hertz, 
                           bands, diadic = True, relative_window_shift = 0.5, 
                           fft_length = -1):
    if type(bands) is list:
        bands = np.array(bands)
    if type(bands) is np.ndarray:
        bands /= fs_hertz
    N = len(x)
    window_duration_samples = np.round(window_duration_seconds * fs_hertz).astype(int)
    window_start = 0
    window_end = window_duration_samples
    window_shift = np.round(window_duration_samples * relative_window_shift).astype(int)
    w = signal.windows.get_window(window_type, window_duration_samples)
    if type(bands) is float:
        bands = int(bands)
    if type(bands) is int:
        n_bands = bands
        bands = []
        if diadic:
            # f1 = 0.25
            # f2 = 0.50
            f = 0
            for k in range(n_bands, 0, -1):
                # bands.append([f1, f2])
                bands.append([f, 2 ** (-k)])
                f = 2 ** (-k)
                # f1 /= 2
                # f2 /= 2
            # bands([0, 2 * f1])
        else:
            f1 = 0.00
            f2 = 0.50 / n_bands
            for k in range(0, n_bands):
                bands.append([f1, f2])
                f1 += 0.50 / n_bands
                f2 += 0.50 / n_bands
        bands = np.array(bands)
    if fft_length == -1:
        fft_length = window_duration_samples
    bands *= fft_length
    bands = np.round(bands).astype(int)
    FBE = []
    while window_end <= N:
        xw = x[window_start : window_end]
        xw *= w
        E = single_window_frequency_bands_energy(xw, bands, fft_length)
        FBE.append(E)
        window_start += window_shift
        window_end += window_shift
    return np.array(FBE)

def single_window_frequency_bands_energy(x, bands_fft_indices, fft_length):
    E = np.zeros(shape = (len(bands_fft_indices), ))
    x_hat = np.fft.fft(x, fft_length)
    for k in range(0, len(bands_fft_indices)):
        frequency_indices = bands_fft_indices[k]
        y = x_hat[frequency_indices[0] : frequency_indices[1], ]
        E[k] = np.sum(np.abs(y) ** 2)
    return E

def visualize_frequency_bands_energy(fbe, fs, T):
    F = fbe.transpose()
    F = F[::-1, :]
    # seaborn.heatmap(F, cmap = 'turbo')
    seaborn.heatmap(F, cmap = 'gray')
    plt.yticks(np.linspace(0, fbe.shape[1], 20), np.round(np.linspace(fs / 2.0, 0, 20)))
    plt.xticks(np.linspace(0, fbe.shape[0], 20), np.round(np.linspace(0, T, 20)))
    plt.xlabel('Tempo (segundos)')
    plt.ylabel('Frequência (hertz)')
    plt.tight_layout()
    plt.show()


def labels(array, label):
    y = []
    
    for i in range (len(array)):
        y.append(label)
    return y



def Energy_loop(array, fs, bands):
    Energy_array = []
    for i in range (len(array)):
        fbe = frequency_bands_energy(array[i], 1, 'hamming', fs, bands, diadic = False, 
                                     relative_window_shift = 1.0, fft_length = -1)
        fbe = np.ravel(fbe)
        Energy_array.append(fbe)
        
    return Energy_array


def Create_set(array1, array2):
    array_conc = np.concatenate((array1, array2), axis=0)
    
    return array_conc



def Energy_EEG_x_y(option1, option2):
    fs = 173.61
    xA, xB, xC, xD, xE = Open_EEG()
    

    y0 = labels(xA, 0)
    y1 = labels(xA, 1)

    
    bands = [
    [0, 3.3],
    [3.3, 6.6],
    [6.6, 9.9],
    [9.9, 13.2],
    [13.2, 16.5],
    [16.5, 19.8],
    [19.8, 23.1],
    [23.1, 26.4],
    [26.4, 29.7],
    [29.7, 33.0]
    ]
    
    # b = 23
    
    # fbe = frequency_bands_energy(xA[b], 1, 'hamming', fs, bands, diadic = False, relative_window_shift = 1.0, fft_length = -1)
    # visualize_frequency_bands_energy(fbe, fs, T)

    xA = Energy_loop(xA, fs, bands)
    xB = Energy_loop(xB, fs, bands)
    xC = Energy_loop(xC, fs, bands)
    xD = Energy_loop(xD, fs, bands)
    xE = Energy_loop(xE, fs, bands)
     
    if option1 == 'A':
        x1 = xA
    if option1 == 'B':
        x1 = xB
    if option1 == 'C':
        x1 = xC
    if option1 == 'D':
        x1 = xD
    if option1 == 'E':
        x1 = xE
    
    if option2 == 'A':
        x2 = xA
    if option2 == 'B':
        x2 = xB
    if option2 == 'C':
        x2 = xC
    if option2 == 'D':
        x2 = xD
    if option2 == 'E':
        x2 = xE
    
    x = Create_set(x1, x2)
    y = Create_set(y0, y1)
    
    return x, y
    
   

def Open_EEG():
    A_path = r'C:\Users\amaur\OneDrive\Área de Trabalho\Eletrônica\Mestrado\Aulas\Projeto 2\sinais\setA'
    B_path = r'C:\Users\amaur\OneDrive\Área de Trabalho\Eletrônica\Mestrado\Aulas\Projeto 2\sinais\setB'
    C_path = r'C:\Users\amaur\OneDrive\Área de Trabalho\Eletrônica\Mestrado\Aulas\Projeto 2\sinais\setC'
    D_path = r'C:\Users\amaur\OneDrive\Área de Trabalho\Eletrônica\Mestrado\Aulas\Projeto 2\sinais\setD'
    E_path = r'C:\Users\amaur\OneDrive\Área de Trabalho\Eletrônica\Mestrado\Aulas\Projeto 2\sinais\setE'
    
    A = read_data(A_path)
    B = read_data(B_path)
    C = read_data(C_path)
    D = read_data(D_path)
    E = read_data(E_path)
    
    
    A = split_to_float(A)
    B = split_to_float(B)
    C = split_to_float(C)
    D = split_to_float(D)
    E = split_to_float(E)
    
    
    
    return A, B, C, D, E
   
    

    
if __name__ == '__main__':
    
    x, y = Energy_EEG_x_y('A', 'D')
    
    
    
    #Aplica one hot encode na variável y
    y = tf.keras.utils.to_categorical(y)
    
    
    
    #Separa o conjunto de dados em dados de treinamento e dados de teste
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    


    #Cria o modelo MLP definindo três camadas e suas características
    model = tf.keras.Sequential([tf.keras.layers.Dense(200, activation='relu', input_shape=(230,)),
                                  tf.keras.layers.Dense(200, activation='relu'),
                                  tf.keras.layers.Dense(2, activation='softmax')])


    #Compila o modelo criado aplicando o otimizador e a função loss
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    #Treina o modelo com os dados de treinamento
    model.fit(x_train, y_train, epochs=100, batch_size=25)

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
    seaborn.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Matriz de Confusão')
    plt.xlabel('Classe Prevista')
    plt.ylabel('Classe Real')
    plt.show()