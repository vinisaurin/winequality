# winequality

a. Como foi a definição da sua estratégia de modelagem?

A fim de explorar o problema da qualidade dos vinhos portugueses do tipo "Vinho Verde", defini como estratégia de modelagem a utilização das Redes Neurais e dos métodos de ensemble mais utilizados no mundo de machine learning atualmente. Entre os métodos de ensemble escolhi os métodos de Bagging: Random Forests e Extra Trees e os métodos de Boosting: XGBoost, Light GBM e CatBoost. 

Além de utilizar diversos métodos de modelagem, também inseri uma diversidade nos dados ao aplicar estes mesmos mecanismos aos dados após a aplicação da técnica de aprendizado não supervisionado PCA. Ao invés de escolher um destes modelos como o melhor, resolvi utilizar a técnica de Stacking para conseguir agregar poder preditivo ao modelo final. 

b. Como foi definida a função de custo utilizada?

A função de custo utilizada foi a Categorical Cross-Entropy, a qual está apresentada abaixo:

<img src="https://latex.codecogs.com/gif.latex?\sum_{c=0}^9&space;y_{o,c}&space;log(p_{o,c})">

Onde y indica 1 se a observação <i>o</i> pertence à classe <i>c</i>, caso contrário indica 0; e <i>p</i> é a probabilidade predita da observação <i>o</i> pertencer à classe <i>c</i>. 

E para os métodos baseados em árvores foi utilizada o critério <i> Gini Impurity </i> descrito abaixo:

<img src="https://latex.codecogs.com/gif.latex?\sum_{i=0}^9&space;f_{i}&space;(1-f_{i})">

Onde <img src="https://latex.codecogs.com/gif.latex?\inline&space;f_{i}"> é a frequência da classe i no nó.

https://latex.codecogs.com/gif.latex?\inline&space;f_{i}

c. Qual foi o critério utilizado na seleção do modelo final?

Como explicado anteriormente, não houve uma seleção de um modelo dado que foi utilizada uma técnica de <i>stacking</i>, a qual é reponsável por aprender a melhor maneira de agregar todos os modelos utilizados. Mas a métrica utilizada para fazer a validação e ajustar os hyperparametros dos modelos foi a <i>accuracy</i> que é a média do número de classificações corretas, entre as diferentes classes. Esta foi a métrica escolhida, pois entendo que não há razões suficientes para utilizar precision ou recall.

d. Qual foi o critério utilizado para validação do modelo? Por que escolheu utilizar este método?

Em razão do número reduzido de observações, foi necessário separar 15% das amostras para o conjunto de teste e o restante foi utilizado no conjunto de treino. Entretanto, na etapa de treino foi utilizada a técnica de K-Fold cross validation, a fim de não reduzir ainda mais a amostra de treino para separar observações para o conjunto de validação.

e. Quais evidências você possui de que seu modelo é suficientemente bom?
