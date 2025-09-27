# Criando uma rede neural básica
Os treinos realizados nesta pasta foram feitos com base no canal <br />

* [SentDex](https://www.youtube.com/@sentdex). <br />


Pequenas modificações foram feitas apenas para estudo e testes, a autoria permanece ao vídeo e livro usado como base também no vídeo .<br><br>




## Algumas formulas utilizadas :
### *SOFTMAX*<BR>

![formula softmax.](https://miro.medium.com/v2/resize:fit:500/0*fbg5QEc2Lv8IIKcq.png "formula softmax.") <br />

* também chamada de saída, vai transformar vetores scores em um vetor de probabilidade, ajuda evitar overflow <br>
 **Usada no trecho** :
 
```python
class Activation_Softmax:
    def foward(self,entradas):
        valor_exp = np.exp(entradas - np.max(entradas, axis=1, keepdims=True))
        probabildades = valor_exp/np.sum
        self.saida = probabildades
```



