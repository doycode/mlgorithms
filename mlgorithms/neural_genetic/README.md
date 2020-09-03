[中文](README_CH.md)
![](https://miro.medium.com/max/1800/1*36MELEhgZsPFuzlZvObnxA.gif)

# NGA: neural-genetic-algorithm

This is an example of optimizing neural network with genetic algorithm. 
We tested NGA and other algorithms(KNN, Logistic Regression, SVM) on the iris dataset. 
First, we scramble the data and train with two thirds of it, and then test on the remaining data.
We execute 10 times for each algorithm and finally average the accuracy, below is the result.

| Algorithm  | Average Accuracy |
| :--- | ---: |
| KNN| 96.8%|
| Logistic Regression|  96.8%|
| SVM| 96.2%|
| NGA|  97.8%|

In fact, although the above experimental conditions are the same, the results are still not convincing, because the parameters of each algorithm are different.
The NGA algorithm and its parameters are introduced below.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Installing

#### Method 1
Clone the project locally and enter the project folder.

```
pip install .
```

#### Method 2
We have also deployed the project to [PyPI](https://pypi.org/project/mlgorithms/), and you can install it anytime, anywhere through the following instruction.

```
pip install mlgorithms
```


## Running the test

If you install successfully, below is the test code for NGA.

```python
from mlgorithms.neural_genetic import NGA
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
in_num = X.shape[1]  # Number of neurons in the input layer(equal to the number of features)
out_num = np.unique(y).shape[0]  # Number of neurons in the output layer(equal to the number of categories)
inst = NGA.NeuralGeneticAlgorithm(in_nodes_num=in_num, out_nodes_num=out_num)  # the two parameters are required
inst.model_fit(X=X, y=y)
inst.fitness_plot()  # Plot the maximum fitness of each generation of genetic algorithm.
predict_label1 = inst.model_predict(X=X, model=None)  # make predictions immediately after fitting

#or you can save model, load model and predict
inst.model_save("./model/test.model")
model = inst.model_load("./model/test.model")
predict_label2 = inst.model_predict(X=X, model=model)
```



## API Reference

```
class mlgorithms.neural_genetic.NeuralGeneticAlgorithm(pop_size=20, 
                                                       gen_iter_num=100, 
											       p_crossover=0.8, 
											       p_mutation=0.05,
												   in_nodes_num=None,
												   hidden_nodes_num=10,
												   out_nodes_num=None,
												   learning_rate=0.01,
												   bias=True,
												   hidden_activ="sigmoid",
												   output_activ="sigmoid",
												   ann_iter_num=1000)
```

## Parameters

| name  | type |required?| default | description |
| :--- | --- | --- |---- | ---: |
| pop_size| int| no|20|Population size of genetic algorithm. |
| gen_iter_num|  int| no| 100|Genetic algorithm iterations.|
| p_crossover|  float| no| 0.8|Cross probability of genetic algorithm.|
| p_mutation|  float| no| 0.05|Mutation probability of genetic algorithm.|
| in_nodes_num|  int| no | None|Number of neurons in the input layer(equal to the number of features). If the parameter is none, the algorithm calculates the parameter automatically based on the input feature data set.|
| hidden_nodes_num|  int| no| None |Number of neurons in the hidden layer. If the parameter is none, the algorithm calculates the parameter automatically.|
| out_nodes_num|  int| no | None|Number of neurons in the output layer(equal to the number of categories). If the parameter is none, the algorithm calculates the parameter automatically based on the input label data set.|
| learning_rate|  float| no| 0.01|Used to update weights.|
| bias|  bool| no| True|Bias unit.|
| hidden_activ |  string| no| "sigmoid"|Activation function of the hidden layer ("sigmoid", "tanh" or "relu").|
| output_activ | string | no | "sigmoid" |Activation function of the output layer ("sigmoid", "tanh" or "relu").|
| ann_iter_num|  int| no| 1000|Neural network iterations.|



## Methods

* **model_fit(self, X, y)**  

      X: ndarray or list, shape=(n_samples, n_features)  
	    
      y: labels, ndarray or list, shape=(n_samples,)  
         
      Returns: None
	
* **model_predict(self, X, model)**  

      X: ndarray or list, shape=(n_samples, n_features)  
         
      model: saved model or None
             When it is none, that means we should make predictions immediately after fitting.  
         
      Returns:  
             y: ndarray, shape=(n_samples).
	        The predicted classes. 
	
* **model_save(self, model_path)**  

      model_path: string or None
              The string include model name, for example: model_path="./model/test.model", when it is None,   model_path="./model/NGA_timestamp.model"  
      
      Returns: None

* **model_load(self, model_path)**  

      model_path: string
              The string include model name, for example: model_path="./model/test.model"  
      Returns:
              model(some parameters)
  
* **fitness_plot(self)**  

      In NGA, fitness function is the prediction accuracy of the model. This method realizes the function of plotting the maximum fitness of each generation of the genetic algorithm.
      
      Returns: None
