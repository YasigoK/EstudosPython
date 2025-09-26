# Código seguindo a playlist de vídeos do canal : @sentdex
# Link : https://youtu.be/Wo5dMEP_BbI?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3

import numpy as np 
import os 
os.system('cls')
np.random.seed(0)

X = np.random.uniform(-5,5,size=(3,4))
#X = 2*np.random.randn(3,4)

inputs = [0 ,2 ,-1 ,3.3 ,-2.7 ,1.1 ,2.2 ,-100]
randomInputs = np.round((np.random.randn(8)*3),1)
print(randomInputs)
output=[]

for i in inputs: # vai percorrer 8 vezes
    output.append(max(0,i))

print(output)


