# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:57:50 2020

@author: yunchengdong
"""


'''
BP神经网路一般都选用二级（3层）网络：
可以证明一个三层网络可以实现以任意精度近似任意连续函数。
'''

'''
在没有偏置量的情况下，输入层神经元个数等于特征数；
在有偏置量的情况下，输入层神经元个数等于特征数+1；
输出层神经元个数等于要分类的个数；

针对三层带偏置的网络:
    假如所有样本有 n_labels 类，每个样本特征数为 n_features，隐藏层神经元个数为 hidden_layer_neural_num，
    则输入层和隐藏层之间的权重矩阵 W1 的shape为 hidden_layer_neural_num x (n_features + 1)
    隐藏层与输出层之间的权重矩阵 W2 的shape为 n_labels x (hidden_layer_neural_num + 1)
    此时输入X的shape为 n_samples x n_features
    那么我们需要对 X 拼接上全为1的列，然后转置，此时X的shape为 (n_features + 1) x n_samples
    W1乘以X得出隐藏层输出hidden_output的shape为 hidden_layer_num x n_samples
    由最终输出final_output = W2 x hidden_output可知，此时hidden_output需要拼接上全为1的行，
    最终得出final_output的shape为 n_labels x n_samples
'''

'''
在确定隐层节点数时必须满足下列条件：
（1）隐层节点数必须小于N-1（其中N为训练样本数），否则，网络模型的系统误差与训练样本
的特性无关而趋于零，即建立的网络模型没有泛化能力，也没有任何实用价值。同理可推得：
输入层的节点数（变量数）必须小于N-1。
(2) 训练样本数必须多于网络模型的连接权数，一般为2~10倍，否则，样本必须分成几部分并
采用“轮流训练”的方法才可能得到可靠的神经网络模型。 
'''

'''
隐藏层的激活函数通常不会选择sigmoid函数？有无依据？
sigmoid一般不用来做多类分类，而是用来做二分类？
'''

import numpy as np

from scipy.stats import truncnorm
from sklearn.preprocessing import OneHotEncoder

#np.vectorize(sigmoid(x))
@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)

@np.vectorize
def relu(x):
    result = x
    result[x < 0] = 0
    return result

#待添加tanh激活函数

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)



class ArtificialNeuralNetwork:
          
    def __init__(self, 
                 in_nodes_num=None,  #输入层神经元个数  number of neurons in input layer
                 out_nodes_num=None,  #输出层神经元个数  number of neurons in output layer
                 hidden_nodes_num=10,  #隐藏层神经元个数  number of neurons in hidden layer
                 learning_rate=0.01,  #学习率，用于更新权重  used to update weights
                 bias=None,  #偏置量  y=wx+b  b is the bias
                 activation_func="sigmoid"  #激活函数  activation function
                ):  

        self.in_nodes_num = in_nodes_num
        self.out_nodes_num = out_nodes_num
        
        self.hidden_nodes_num = hidden_nodes_num
            
        self.learning_rate = learning_rate 
        self.bias = bias
        #self.create_weight_matrices()
        
        self.activation_func = activation_func
    
    
    def create_weight_matrices(self):
        """ A method to initialize the weight matrices of the neural network with optional bias nodes.
        
            What is the benefit of the truncated normal distribution in initializing weights in a neural network?
    
            I think its about saturation of the neurons. Think about you have an activation function like sigmoid.
            If your weight val gets value >= 2 or <=-2 your neuron will not learn. So, if you truncate your normal 
            distribution you will not have this issue(at least from the initialization) based on your variance. I 
            think thats why, its better to use truncated normal in general.
        """
        
        bias_node = 1 if self.bias else 0
        
        self.rad1 = 1 / np.sqrt(self.in_nodes_num + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-self.rad1, upp=self.rad1)
        self.weights_hidden_in = X.rvs((self.hidden_nodes_num, 
                                       self.in_nodes_num + bias_node))

        self.rad2 = 1 / np.sqrt(self.hidden_nodes_num + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-self.rad2, upp=self.rad2)
        self.weights_hidden_out = X.rvs((self.out_nodes_num, 
                                        self.hidden_nodes_num + bias_node))
        
        
    def train(self, input_vector=None, target_vector=None):
        '''BP网络训练
        
        Args:
            input_vector: 输入样本特征数据  shape: (n_samples, n_features)  格式：ndarray
            target_vector: 输入样本标签  shape: (n_samples, )  格式：ndarray
        '''
        
        if self.bias:
            # adding bias node to the end of the inpuy_vector
            row_num, col_num = np.shape(input_vector)
            col_ones = np.ones((row_num, 1))
            input_vector = np.concatenate((input_vector, col_ones), axis=1)  #添加全1列
                                    
            
        input_vector = input_vector.T
        
        #one-hot编码
        target_vector = target_vector.reshape(-1, 1)
        one_hot_encoder = OneHotEncoder(sparse=False)
        one_hot_encoded = one_hot_encoder.fit_transform(target_vector)
        self.target_vec = one_hot_encoded.T  #转置

        
        output_vector1 = np.dot(self.weights_hidden_in, input_vector)
        
        if "relu" == self.activation_func:
            output_vector_hidden = relu(output_vector1)
        else:
            output_vector_hidden = sigmoid(output_vector1)
        
        if self.bias:
            row_num, col_num = np.shape(output_vector_hidden)
            row_ones = np.ones((1, col_num))
            output_vector_hidden = np.concatenate((output_vector_hidden, row_ones), axis=0)  #添加全1行
        
        
        output_vector2 = np.dot(self.weights_hidden_out, output_vector_hidden)
        
        if "relu" == self.activation_func:
            output_vector_network = relu(output_vector2)
        else:
            output_vector_network = sigmoid(output_vector2)
        
        output_errors = self.target_vec - output_vector_network
        # update the weights:
        tmp = output_errors * output_vector_network * (1.0 - output_vector_network)     
        tmp = self.learning_rate  * np.dot(tmp, output_vector_hidden.T)
        self.weights_hidden_out += tmp


        # calculate hidden errors:
        hidden_errors = np.dot(self.weights_hidden_out.T, output_errors)
        # update the weights:
        tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)
        if self.bias:
            x = np.dot(tmp, input_vector.T)[:-1,:]     # ???? last element cut off, ???
        else:
            x = np.dot(tmp, input_vector.T)
        self.weights_hidden_in += self.learning_rate * x
        
       
    
    def predict(self, input_vector=None):
        '''预测
        
        Args:
            input_vector: 输入样本特征数据  shape: (n_samples, n_features)  格式：ndarray
        '''
        
        if self.bias:
            # adding bias node to the end of the inpuy_vector
            row_num, col_num = np.shape(input_vector)
            col_ones = np.ones((row_num, 1))
            input_vector = np.concatenate((input_vector, col_ones), axis=1)  #添加全1列
            
        input_vector = input_vector.T

        output_vector = np.dot(self.weights_hidden_in, input_vector)
        if "relu" == self.activation_func:
            output_vector = relu(output_vector)
        else:
            output_vector = sigmoid(output_vector)
        
        if self.bias:
            row_num, col_num = np.shape(output_vector)
            row_ones = np.ones((1, col_num))
            output_vector = np.concatenate((output_vector, row_ones), axis=0)  #添加全1行
            

        output_vector = np.dot(self.weights_hidden_out, output_vector)
        
        if "relu" == self.activation_func:
            output_vector = relu(output_vector)
        else:
            output_vector = sigmoid(output_vector)
        
        output_label = np.argmax(output_vector, axis=0)
    
        return output_vector, output_label