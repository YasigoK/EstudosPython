# Código seguindo a playlist de vídeos do canal : @sentdex
# Link : https://youtu.be/Wo5dMEP_BbI?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3

import numpy as np 
import os 
os.system('cls')

#Achei melhor usanr np rand pra ficar mais prático e fácil de testar 

#entradas = [1.2,2.3,3.4,4.5]
entradas1 =np.round(np.random.rand(3,4),3)

#pesos = [ 0.1,1.2,2.3,-2.2]

pesos1 = np.round(np.random.rand(3,4),3)

#bias = 2.0
bias1 = np.round(np.random.rand(3),3)

print(f"entradas : {entradas1}")
print(f"\npesos : {pesos1}")
print(f"\nbias = {bias1}")

#precisa transformar uma em transposta para poder somar as matrizes 
#saida1 = np.dot(pesos1,np.array(entradas1).T) + bias1
saida1 = np.dot(entradas1,np.array(pesos1).T) + bias1

print(f"\n\nsaida = {np.round(saida1,1)}")

print(f"\n\n===============TESTANDO OUTROS VALORES=========================\n")

pesos2 = np.round(np.random.uniform(-4,4,size=(3,3)),2)
print(f"\n pesos = {pesos2}")

bias2 = np.round(np.random.uniform(-4,4,size=3),1)
print(f"\n bias = {bias2}")

saida2=np.dot(saida1,np.array(pesos2).T)+bias2

print(f"\n saida 2 = {saida2}")

