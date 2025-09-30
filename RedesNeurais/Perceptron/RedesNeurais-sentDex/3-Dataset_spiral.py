import numpy as np
import os 
import nnfs 
from nnfs.datasets import spiral_data # importando uma série de dados que serve como base para trinamento, é um monte de ponto espalhado em um plano, bem confuso a olho nu


os.system('cls')
os.system('clear')
nnfs.init()

# X em maiusculo serve para caracterizar a entrada, uma matriz 2d 
# y será a saída, vetor 1d 



class Camada_Densa: #nome bizarro 
    def __init__(self, n_input, n_neurons):
        self.peso = 0.1*np.random.randn (n_input,n_neurons)
        self.bias = np.zeros((1,n_neurons))

    def foward(self, entradas):
        self.output = np.dot(entradas, self.peso) + self.bias

class Ativando_ReLU:# substitui entradas negativas por 0
    def foward(self,entradas):
        self.output = np.maximum(0,entradas) # 

class Ativando_Softmax:
    def foward(self,entradas):
        valor_exp = np.exp(entradas - np.max(entradas, axis=1, keepdims=True)) # o trecho do np.max serve para o valor maximo de cada entrada, onde o axis1= serve pra indicar todas as colunas e o keepdims server para garantir que o  resultado mantenha  a dimensão original

        probabildades = valor_exp/np.sum(valor_exp, axis=1, keepdims=True) # continuando com a formula padrão do softmax

        self.output = probabildades
        
class Loss:
    def calcula(self,output,y0):
        sample_loss = self.foward(output,y)
        data_loss = np.mean(sample_loss)
        return data_loss
    
class Entropia_cruzada(Loss):
    def foward(self,y_pred,y_true):
        amostras = len(y_pred)
        y_pred_reduz = np.clip(y_pred,1e-7,1e-7)#reponsável para não dar valoreos infinitos, pq caso de 0 pode ocorrer um erro de valor infinito 
        if len(y_true.shape)==1:
            correcao_de_confidencias = y_pred_reduz[range(amostras),y_true] # aqui é a seleção de itens das amostras,o  range (amostras) vai primeiro pegar todas as amostras disponíveis, e o ,y_true serve apra filtra quais querem pegar, pegando um valor por linhas
        elif len(y_true.shape) == 2 :
            correcao_de_confidencias = np.sum(y_pred_reduz * y_true, axis = 10)

        log_Negativo = -np.log(correcao_de_confidencias)
        return log_Negativo

X,y = spiral_data(samples = 100,classes = 3) # gerando 100 unidades para 3 tipos de classes 

camada1 = Camada_Densa(2,5)
ativacao1 = Ativando_ReLU()

camada2 = Camada_Densa(5,3)
ativacao2 = Ativando_Softmax()

camada1.foward(X)
ativacao1.foward(camada1.output)

camada2.foward(ativacao1.output)
ativacao2.foward(camada2.output)

print(f"saida : \n{ativacao2.output[:8]}")

funcao_Loss = Entropia_cruzada()
loss = funcao_Loss.calcula(ativacao2.output,y)

print (f"\nloss : {loss}" )
