---
title: Evolução Diferencial vs Jade vs Algoritmo Genético
date: 2022-03-02
---

## Motivação

Na disciplina de Inteligência Artificial e Aprendizagem de Máquina ministrada pelo professor Leandro Coelho no curso de Engenharia Elétrica da UFPR, foi requisitada a comparação entre o funcionamento do algoritmo genético e a evolução diferencial. Aproveitando o incentivo da disciplina também comparei estes dois algoritmos, implementados por mim, com a [evolução diferencial](https://link.springer.com/article/10.1023/A:1008202821328) presente no [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html) e com uma implementação da [jade](https://ieeexplore.ieee.org/document/5208221).

Todos os códigos produzidos por mim podem ser encontrados em [AIML](https://github.com/CaioMizerkowski/AIML), construídos em Python e utilizando principalmente a biblioteca [numpy](https://numpy.org/) e com o uso do [pandas](https://pandas.pydata.org/) para o describe final dos dados. O material aqui expresso é em grande parte derivado do aprendizado realizado na disciplina, sendo qualquer equívoco expresso aqui de responsabilidade totalmente minha.

## Algoritmo Genético com números com ponto flutuantes

Algoritmos genéticos constituem uma família de métodos para realizar a otimização de funções, encontrar seus pontos mínimos ou máximos dentro de uma série de restrições. Eles são baseados na teoria da evolução e apresentam uma versão simplificada, mas bastante potente, dos genes através de valores arrays de valores numéricos que codificam os parâmetros a serem otimizados. Estes valores podem ser codificados em valores binários, em valores inteiros (usados para problemas de permutação) ou em valores reais (ponto flutuante). Em razão dos problemas apresentados para a otimização terem seus parâmetros definidos dentre os números reais, foi implementada a última opção.

Os algoritmos genéticos começam com a geração de uma população, na qual cada indivíduo possui um conjunto aleatório de genes. Três processos, derivados da teoria da evolução, são realizados nesta população: Mutação, recombinação e seleção. O seguinte trecho de código mostra esse processo, no qual a população *X* possui seus indivíduos _x_ sofrendo estes três processos de forma a melhorar seu fitness.

```python
def genetic_algorithm(self):
    self.X_vector = [self.X]
    self._g = 0
    while self._g < self._G:
        X_new = np.zeros(self._shape)
        self.best_x = self.X[np.array(map(self._fitness, self.X)).argmin()]
        for idx, x in enumerate(self.X):
            v = self._mutation(x)
            v = self._crossover(v)
            X_new[idx] = self._selection(x, v)
        self.X = X_new
        self.X_vector.append(self.X)
        self._g += 1
```

## Evolução diferencial

Uma alteração nos algoritmos genéticos clássicos, a evolução diferencial se distancia um pouco da analogia evolucionária clássica de cruzamento somente entre dois indivíduos da população. Sendo o resultado do processo de mutação um cruzamento de três ou mais indivíduos, mediado por um fator de mutação _mr_.

Diferentes estratégias de mutação existem, mas a mais comum é a utilizada neste trabalho foi a DE/rand/1/bin, no qual o operador de mutação é:

\\[ V_{i} = X_{r^{i}_{1}} + mr*(X_{r^{i}_{2}}-X_{r^{i}_{3}}) \\]

Sendo o fator diferencial, que da o nome ao algoritmo, a diferença $$X_{r^{i}_{2}}-X_{r^{i}_{3}}$$, pois quanto maior a proximidade de ambos, menor o seu efeito será e a busca se tornará mais regional, ao invés de ser global.

Outro parâmetro do algoritmo é o _cr_, que determina qual a probabilidade de recombinação entre os genes de *X* e de *V*. No caso clássico da Evolução Diferencial estes valores são ajustados a mão, enquanto que técnicas mais avançadas como a JADE permitem que estes parâmetros sejam ajustados durante a execução.

 
## JADE com Arquivo
 
Como diferentes valores de _mr_ e _cr_ podem gerar resultados bastante discrepantes nos mínimos encontrados e como o ajuste destes valores na mão pelo cientista responsável acaba por ser um processo repetitivo, variantes que permitem o ajustes destes parâmetros durante a execução do programa são bastantes estudadas na literatura e usadas na prática.
 
Uma variante popular e poderosa é a JADE, no qual a cada iteração _mr_ e _cr_ individuais são ajustados conforme os que tiveram os melhores resultados na iteração anterior.
 
O ajuste dos _mr_ se dá por meio de uma distribuição normal cujo centro é definido pelos _mr_ de sucesso na etapa anterior. O análogo ocorreu para o _cr_, mas utilizando-se uma distribuição de Cauchy.
 
Outra modificação da JADE é o operador de mutação utilizado, conhecido como DE/current-to-pbest/1:
 
\\[ V_{i} = X_{i} + mr_{i}*(X^{P}_{best}-X_{i}) + mr*(X_{r^{i}_{1}}-X_{r^{i}_{2}}) \\]
 
No qual $$X^{P}_{best}$$ é um dos p% melhores indivíduos da população, evitando que o algoritmo seja muito guloso na sua busca por mínimos locais.
 
## Resultados
 
Ainda em construção! Esperando validação dos resultados pelo professor.
