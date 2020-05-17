# winequality

a. Como foi a definição da sua estratégia de modelagem?

A fim de explorar o problema da qualidade dos vinhos portugueses do tipo "Vinho Verde", defini como estratégia de modelagem a utilização das Redes Neurais e dos métodos de ensemble mais utilizados no mundo de machine learning atualmente. Entre os métodos de ensemble escolhi os métodos de Bagging: Random Forests e Extra Trees e os métodos de Boosting: XGBoost, Light GBM e CatBoost. 

b. Como foi definida a função de custo utilizada?

A função de custo utilizada foi a Categorical Cross-Entropy, a qual está apresentada abaixo:

<img src="https://latex.codecogs.com/gif.latex?\sum_{c=0}^9&space;y_{o,c}&space;log(p_{o,c})">

Onde y indica 1 se a observação <i>o</i> pertence à classe <i>c</i>, caso contrário indica 0; e <i>p</i> é a probabilidade predita da observação <i>o</i> pertencer à classe <i>c</i>. 

E para os métodos baseados em árvores foi utilizado o critério <i> Gini Impurity </i> descrito abaixo:

<img src="https://latex.codecogs.com/gif.latex?\sum_{i=0}^9&space;f_{i}&space;(1-f_{i})">

Onde <img src="https://latex.codecogs.com/gif.latex?\inline&space;f_{i}"> é a frequência da classe <i>i</i> no nó.

c. Qual foi o critério utilizado na seleção do modelo final?

O critério utilizado na seleção do modelo final foi o f1-score, que leva em consideração as métricas de precision e recall. Quanto maior esse indicador, melhor é o modelo.

d. Qual foi o critério utilizado para validação do modelo? Por que escolheu utilizar este método?

A métrica utilizada para fazer a validação e ajustar os hyperparametros dos modelos foi a <i>accuracy</i> que é a média do número de classificações corretas, entre as diferentes classes. Em razão do número reduzido de observações, foi necessário separar 15% das amostras para o conjunto de teste e o restante foi utilizado no conjunto de treino. Entretanto, na etapa de treino foi utilizada a técnica de K-Fold cross validation, a fim de não reduzir ainda mais a amostra de treino para separar observações para o conjunto de validação.

e. Quais evidências você possui de que seu modelo é suficientemente bom?

A técnica de CatBoost teve os melhores resultados neste problema. Entendo que o modelo é suficientemente bom, pois possui níveis de precision e recall acima de 70% nas classes com maior número de observação (quality igual a 5 ou 6). Nas outras classes, estes níveis não são tão bons, pois o número de observações é muito inferior, fazendo com que o modelo não tivesse dados o suficiente para aprender sobre estas classes.
