# winequality

a. Como foi a definição da sua estratégia de modelagem?

A fim de explorar o problema da qualidade dos vinhos portugueses do tipo "Vinho Verde", defini como estratégia de modelagem a utilização das Redes Neurais e dos métodos de ensemble mais utilizados no mundo de machine learning atualmente. Entre os métodos de ensemble escolhi os métodos de Bagging: Random Forests e Extra Trees e os métodos de Boosting: XGBoost, Light GBM e CatBoost. 

Além de utilizar diversos métodos de modelagem, também inseri uma diversidade nos dados ao aplicar estes mesmos mecanismos aos dados após a aplicação da técnica de aprendizado não supervisionado PCA. Ao invés de escolher um destes modelos como o melhor, resolvi utilizar a técnica de Stacking para conseguir agregar poder preditivo ao modelo final. 

b. Como foi definida a função de custo utilizada?
A função de custo utilizada foi a Categorical Cross-Entropy, a qual está apresentada abaixo:
{-\sum_{c=1}^{M}}

<img src="https://latex.codecogs.com/gif.latex?CE&amp;space;=&amp;space;-log\left&amp;space;(&amp;space;\frac{e^{s_{p}}}{\sum_{j}^{C}&amp;space;e^{s_{j}}}&amp;space;\right&amp;space;)" title="CE = -log\left ( \frac{e^{s_{p}}}{\sum_{j}^{C} e^{s_{j}}} \right )">


c. Qual foi o critério utilizado na seleção do modelo final?

d. Qual foi o critério utilizado para validação do modelo? Por que escolheu utilizar este método?
e. Quais evidências você possui de que seu modelo é suficientemente bom?
