import numpy as np
import os 
import nnfs 

os.system('cls')
nnfs.init()

X = np.round(np.random.uniform(-5,5, size=(3,4)),1) 
print(f"entradas : \n {X}")

class Camada_Densa: #nome bizarro 
    def __init__(self, n_input, n_neurons):
        self.peso = 0.1*np.random.randn (n_input,n_neurons)
        self.bias = np.zeros((1,n_neurons))

    def atualiza(self, entradas):
        print("===========================================")
        self.output = np.dot(entradas, self.peso) + self.bias
        print(f"saida : \n {self.output}")

class Ativando_ReLU:# substitui entradas negativas por 0
    def atualiza2(self,entradas):
        self.output = np.maximum(0,entradas) # 


    

camada1 = Camada_Densa(4,5)
camada2 = Camada_Densa(5,2)

camada1.atualiza(X)
camada2.atualiza(camada1.output)
    