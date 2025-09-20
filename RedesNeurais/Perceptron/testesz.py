#aqui são só alguns testes de codigo em python que eu vou fazer, não sao relevantes pra nenhum projeto
# isso vem muito do pq eu to aprendendo python ao mesmo tempo que estudo redes neurais


import numpy as np
teste = 10*np.random.rand(5)
bias = np.random.rand()
testint = teste.astype(int)
print(testint + bias)



numeros =[1,14,513,63,47,234]
repet = 30

for alele in range(repet):
    print(alele)

for a,b in enumerate(numeros):
    print(f"posicao [{a}] com o valor :{b}")

