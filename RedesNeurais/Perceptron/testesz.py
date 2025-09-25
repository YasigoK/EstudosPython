#aqui são só alguns testes de codigo em python que eu vou fazer, não sao relevantes pra nenhum projeto
# isso vem muito do pq eu to aprendendo python ao mesmo tempo que estudo redes neurais
import numpy as np 

from perceptron1 import Perceptron

import numpy as np
teste = 10*np.random.rand(5)
bias = np.random.rand()
testint = teste.astype(int)
print(testint + bias)

arrayTeste1 = np.array([12,4,5])
copiaArray = arrayTeste1
copiaArray2 = arrayTeste1.copy()

for a,b in enumerate(arrayTeste1):
    print(arrayTeste1)
    print(copiaArray)
    print(copiaArray2)
    print("\n\n")
    arrayTeste1*=3



numeros =[1,14,513,63,47,234]
repet = 3
for alele in range(repet):
    print(alele)

for a,b in enumerate(numeros):
    print(f"posicao [{a}] com o valor :{b}")

