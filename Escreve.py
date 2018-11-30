#Escreve
def contagem(x,y): # nome do arquivo, tamanho da contagem
    arq = open(x,"w")
    i = 0
    while (i < y):
        x = str(i)
        arq.write(x + "\n")
        i += 1

    arq.close()

#Lê e adiciona 1
def leesoma(x,c): #nome do arquivo, valor da soma
    ler = open(x, "r")

    i = 0
    t = 0
    j = [0,0,0,0,0,0,0,0,0,0]
    y = ler.read(20)
    while (i < 20):         #ARRUMAR PARA SER QUALQUER TAMANHO DE DOCUMENTO
        j[t] = y[i]
        t += 1
        i += 2              #Pula os \n

    i = 0
    j = list(map(int,j))   #converte a lista em um vetor de ints
    while(i < 10):
        j[i] += c
        i += 1
    
    
    
    return j

def escreve(x, y):
    i = 0
    arq = open(x,"w")
    while (i < 10):
        y[i] = str(y[i])
        arq.write(y[i] + "\n")
        i += 1

    arq.close()

contagem("TXT.txt",20)
x = leesoma("TXT.txt",1)
print(x)
escreve("TXT1.txt", x)



 
