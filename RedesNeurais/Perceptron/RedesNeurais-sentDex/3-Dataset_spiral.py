import numpy as np
import os 
import nnfs 
from nnfs.datasets import spiral_data # importando uma série de dados que serve como base para trinamento, é um monte de ponto espalhado em um plano, bem confuso a olho nu


os.system('cls')
nnfs.init()

# X em maiusculo serve para caracterizar a entrada, uma matriz 2d 
# y será a saída, vetor 1d 
X,y = spiral_data(100,3) # gerando 100 unidades para 3 tipos de classes 


class Camada_Densa: #nome bizarro 
    def __init__(self, n_input, n_neurons):
        self.peso = 0.1*np.random.randn (n_input,n_neurons)
        self.bias = np.zeros((1,n_neurons))

    def atualiza(self, entradas):
        self.output = np.dot(entradas, self.peso) + self.bias

class Ativando_ReLU:# substitui entradas negativas por 0
    def atualiza2(self,entradas):
        self.output = np.maximum(0,entradas) # 


camada1 = Camada_Densa(2,5)# se torna 2 pq no dataset usa-se 2 tipos de entradas (XY), sendoassim precisa ter o mesmo tamanho para não dar problema com os pesos 

ativando1 = Ativando_ReLU()# criando uma var que vai ser usada mais tarde com mo propósito de remover numeros negativos 
camada1.atualiza(X) # realizando processo de treinamento, atualizando o valor de entrada baseando-se nos pesos e bias 
ativando1.atualiza2(camada1.output) # resumidamente pegou os valores gerados da saida da camada1 e posteriormente transoformou todos aqueles negativos em 0 
print(ativando1.output)
    