import numpy as np

class Perceptron:
    def __init__(self,num_entradas):
        self.pesos = np.random.rand(num_entradas)
        self.bias = np.random.rand()
 
    def prever (self, entradas):
        #Primeiro vai calcular a soma ponderada usando numpy
        soma_ponderada = np.dot(self.pesos, entradas) + self.bias

        #tomada de decisão basica prevendo erros
        if soma_ponderada>0:
            return 1
        else:
            return 0
        
    def treinar(self, entradas, valor_esperado, taxa_de_aprendizagem):
        #Primeiramente é preciso obter a previsão do perceptron 
        previsao = self.prever(entradas) # vai pegar o valor da entrada e testar com o valor esperado

        # calcular o erro
        erro = valor_esperado - previsao # como o nome sugere, vai pegar o resultado da ultima operação de comparação e subtrair com o valor esperado 

        for i, peso in enumerate(self.pesos):
            #um looping que vai percorrer cada elemento da lista que foi criada la no começo
            self.pesos[i] = self.pesos[i] + (erro * entradas[i] * taxa_de_aprendizagem)

            # por fim atualizando o bias 
            self.bias+= erro * taxa_de_aprendizagem
        
    