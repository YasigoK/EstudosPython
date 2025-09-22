import numpy as np
import os # mais pra limpar terminal

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
        
    def treinar(self, entradas, valor_esperado, taentradas_logicasa_de_aprendizagem):
        #Primeiramente é preciso obter a previsão do perceptron 
        previsao = self.prever(entradas) # vai pegar o valor da entrada e testar com o valor esperado

        # calcular o erro
        erro = valor_esperado - previsao # como o nome sugere, vai pegar o resultado da ultima operação de comparação e subtrair com o valor esperado 

        for i, peso in enumerate(self.pesos):
            #um looping que vai percorrer cada elemento da lista que foi criada la no começo
            self.pesos[i] = self.pesos[i] + (erro * entradas[i] * taentradas_logicasa_de_aprendizagem)

        # por fim atualizando o bias 
        self.bias+= erro * taentradas_logicasa_de_aprendizagem
        return erro
        

#Treinando porta lógica and e or
#entradas
entradas_logicas = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
#saidas
saidas_and = np.array([
    0,
    0,
    0,
    1
])

saidas_or = np.array([
    0,
    1,
    1,
    1
])
os.system("cls")
os.system("clear")
print("|Digite qual porta deseja treinar :")
print("|    1  - and    ||    2 - or    |")
escolha = input("| Digite sua escolha :")

#usando dicionário ao invés de if else (treinando mesmo)
escolhasPossiveis ={
    '1': {'nome':'And','saidas':saidas_and},
    '2': {'nome':'Or' ,'saidas':saidas_or}
}

porta_escolhida = escolhasPossiveis.get(escolha)
#se digitar um valor diferente de 1 ou 2
if porta_escolhida is None:
    print("Opção inválida. O programa será encerrado.")
    exit()

porta = porta_escolhida['saidas']
nome_porta = porta_escolhida['nome']

#criando uma instancia do perceptron 
meu_perceptron = Perceptron(2) # passando o numero 2 pq temos duas colunas no entradas_logicas
pesos_iniciais = meu_perceptron.pesos.copy()
bias_iniciais = meu_perceptron.bias

#Criando mais alguns dados que serão usados para o looping de treinamento
taentradas_logicasa_de_aprendizado = 0.1
num_epocas = 100
#limpando terminal antes dos prints marotos


print(f"|-=-=-=-= Treinando porta {nome_porta} =-=-=-=-=- |")
for epoca in range(num_epocas):
    print(f"\n-=-=-=repeticao {epoca + 1 } no total de {num_epocas}=--=-=-")
    erro_totales = 0
    
    for entradas, valor_esperado in zip(entradas_logicas,porta):
        pesos_anteriores = meu_perceptron.pesos.copy()
        bias_anteriores = meu_perceptron.bias
        erro = meu_perceptron.treinar(entradas,valor_esperado, taentradas_logicasa_de_aprendizado)

        erro_totales +=abs(erro) # vai somar o valor absoluto da var erro, para verificar se o erro total é zero 

        print(f"  Entradas: {entradas} | Esperado: {valor_esperado} | Erro: {erro}")
        print(f"    Pesos antes : {np.round(pesos_anteriores, 4)} -> Pesos depois : {np.round(meu_perceptron.pesos, 4)}")
        print(f"    Viés antes  : {round(bias_anteriores, 4)}         -> Viés depois  : {round(meu_perceptron.bias, 4)}")

    if erro_totales==0: #ou seja o modelo acertou e n encontrou mais nenhum erro 
        print("Treinamento concluido")
        break
        
print("\n\n -=-=-=treinamento finalizeido=-=-=-")
print(f"porta : {nome_porta}")
print(f"pesos iniciais : {pesos_iniciais}")
print("pesos finais : " , meu_perceptron.pesos)
print(f"vies inicial :  {round(bias_iniciais, 5)}")
print(f"vies final  : {round(meu_perceptron.bias,5)}")
print(f"Num total de épocas {epoca+1}")