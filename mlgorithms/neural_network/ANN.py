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
sigmoid一般不用来做多类分类，而是用来做二分类？(看编码方式，onehot编码每个类别对应位置为0或1)
'''

'''
第k层神经元的误差项是由第k+1层的误差项乘以第k+1层的权重，再乘以第k层激活函数的导数得到
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
    if x < 0:
        x = 0
    
    return x

@np.vectorize
def relu_deriv(x):
    if x <= 0:
        x = 0
    else:
        x = 1
    
    return x

#tanh激活函数
@np.vectorize
def tanh(x):
    return (np.e ** x - np.e ** -x) / (np.e ** x + np.e ** -x)

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)



class ArtificialNeuralNetwork:
          
    def __init__(self, 
                 in_nodes_num=None,  #输入层神经元个数  number of neurons in input layer
                 out_nodes_num=None,  #输出层神经元个数  number of neurons in output layer
                 hidden_nodes_num=None,  #隐藏层神经元个数  number of neurons in hidden layer
                 learning_rate=0.01,  #学习率，用于更新权重  used to update weights
                 bias=True,  #偏置量  y=wx+b  b is the bias
                 hidden_activ="sigmoid",
                 output_activ="sigmoid",
                ):  

        self.in_nodes_num = in_nodes_num
        self.out_nodes_num = out_nodes_num
        
        self.hidden_nodes_num = hidden_nodes_num
            
        self.learning_rate = learning_rate 
        self.bias = bias
        #self.create_weight_matrices()
        
        self.hidden_activ = hidden_activ
        self.output_activ = output_activ
        
        
    def auto_para_base_on_train_data(self, X, y):
        try:
            X = np.array(X)
            y = np.array(y)
            
            assert 2 == X.ndim
            assert 1 == y.ndim
            
            self.in_nodes_num = X.shape[1]
            self.out_nodes_num = np.unique(y).shape[0]
            if self.hidden_nodes_num is None:
                self.hidden_nodes_num = int(
                    X.shape[0] / (np.random.uniform(2, 10)*(self.in_nodes_num+self.out_nodes_num)))
        except Exception as e:
            print(e)
    
    
    def init_weights(self):
        """ 权重初始化：
        用截断正态分布初始化权重的好处是，想象一下sigmoid激活函数，如果权重过大或过小，
        激活函数值趋于饱和化，神经元将无法再学习，但截断正态分布初始化权重则能相对避免
        这种情况。
        
            A method to initialize the weight matrices of the neural network with optional bias nodes.
        
            What is the benefit of the truncated normal distribution in initializing weights in a neural network?
    
            I think its about saturation of the neurons. Think about you have an activation function like sigmoid.
            If your weight val gets value >= 2 or <=-2 your neuron will not learn. So, if you truncate your normal 
            distribution you will not have this issue(at least from the initialization) based on your variance. I 
            think thats why, its better to use truncated normal in general.
        """
        
        try:
            bias_node = 1 if self.bias else 0
            
            if self.in_nodes_num is None or self.out_nodes_num is None:
                print("Please check the input nodes num or hidden nodes num or output nodes num!")
        
            self.rad1 = 1 / np.sqrt(self.in_nodes_num + bias_node)
            X = truncated_normal(mean=0, sd=1, low=-self.rad1, upp=self.rad1)
            self.weights_hidden_in = X.rvs((self.hidden_nodes_num, 
                                           self.in_nodes_num + bias_node))
    
            self.rad2 = 1 / np.sqrt(self.hidden_nodes_num + bias_node)
            X = truncated_normal(mean=0, sd=1, low=-self.rad2, upp=self.rad2)
            self.weights_hidden_out = X.rvs((self.out_nodes_num, 
                                            self.hidden_nodes_num + bias_node))
        except Exception as e:
            print(e)
        
        
    def train(self, input_vector=None, target_vector=None):
        '''BP网络训练
        
        Args:
            input_vector: 输入样本特征数据  shape: (n_samples, n_features)  格式：ndarray
            target_vector: 输入样本标签  shape: (n_samples, )  格式：ndarray
        '''
        try:
            if self.bias:
                # adding bias node to the end of the input_vector
                feature_num = input_vector.shape[1]
                
                row_num, col_num = np.shape(input_vector)
                col_ones = np.ones((row_num, 1))
                input_vector = np.concatenate((input_vector, col_ones), axis=1)  #添加全1列
                
            input_vector = np.array(input_vector)
            target_vector = np.array(target_vector)
            
            assert 2 == input_vector.ndim
            assert 1 == target_vector.ndim
            
            if self.in_nodes_num != feature_num:
                self.in_nodes_num = feature_num #输入层神经元个数等于特征数目
                    
            class_num = np.unique(target_vector).shape[0]
                
            if self.out_nodes_num != class_num:
                self.out_nodes_num = class_num #输出神经元个数等于类别数目
            
            input_vector = input_vector.T
            #one-hot编码
            target_vector = target_vector.reshape(-1, 1)
            one_hot_encoder = OneHotEncoder(sparse=False)
            one_hot_encoded = one_hot_encoder.fit_transform(target_vector)
            self.target_vec = one_hot_encoded.T  #转置
            
            output_vector1 = np.dot(self.weights_hidden_in, input_vector)
            
            if "relu" == self.hidden_activ:
                output_vector_hidden = relu(output_vector1)
            elif "tanh" == self.hidden_activ:
                output_vector_hidden = tanh(output_vector1)
            else:
                output_vector_hidden = sigmoid(output_vector1)
                
            if self.bias:
                row_num, col_num = np.shape(output_vector_hidden)
                row_ones = np.ones((1, col_num))
                output_vector_hidden = np.concatenate((output_vector_hidden, row_ones), axis=0)  #添加全1行
                
            output_vector2 = np.dot(self.weights_hidden_out, output_vector_hidden)
        
            if "relu" == self.output_activ:
                output_vector_network = relu(output_vector2)
            elif "tanh" == self.output_activ:
                output_vector_network = tanh(output_vector2)
            else:
                output_vector_network = sigmoid(output_vector2)
            
            output_errors = -self.target_vec + output_vector_network
            # update the weights:
            if "relu" == self.output_activ: #如果输出层激活函数是relu：
                relu_d = relu_deriv(output_vector_network)
                tmp = output_errors * relu_d
            elif "tanh" == self.output_activ:
                tmp = output_errors * (1 - output_vector_network * output_vector_network)
            else:
                tmp = output_errors * output_vector_network * (1.0 - output_vector_network)
            
            tmp = self.learning_rate  * np.dot(tmp, output_vector_hidden.T)
            self.weights_hidden_out -= tmp
            
            # calculate hidden errors:
            # 第k层神经元的误差项是由第k+1层的误差项乘以第k+1层的权重，再乘以第k层激活函数的导数得到
            hidden_errors = np.dot(self.weights_hidden_out.T, output_errors)
            # update the weights:
            if "relu" == self.hidden_activ:
                relu_d = relu_deriv(output_vector_hidden)
                tmp = hidden_errors * relu_d
            elif "tanh" == self.hidden_activ:
                tmp = hidden_errors * (1 - output_vector_hidden * output_vector_hidden)
            else:
                tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)
            if self.bias:
                x = np.dot(tmp, input_vector.T)[:-1,:]
            else:
                x = np.dot(tmp, input_vector.T)
            self.weights_hidden_in -= self.learning_rate * x
            
        except Exception as e:
            print(e)
        
       
    
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
        if "relu" == self.hidden_activ:
            output_vector = relu(output_vector)
        elif "tanh" == self.hidden_activ:
            output_vector = tanh(output_vector)
        else:
            output_vector = sigmoid(output_vector)
        
        if self.bias:
            row_num, col_num = np.shape(output_vector)
            row_ones = np.ones((1, col_num))
            output_vector = np.concatenate((output_vector, row_ones), axis=0)  #添加全1行
            

        output_vector = np.dot(self.weights_hidden_out, output_vector)
        
        if "relu" == self.output_activ:
            output_vector = relu(output_vector)
        elif "tanh" == self.output_activ:
            output_vector = tanh(output_vector)
        else:
            output_vector = sigmoid(output_vector)
        
        output_label = np.argmax(output_vector, axis=0)
    
        return output_vector, output_label
    
    
def main():
    from sklearn.datasets import load_iris
    iris = load_iris()
    X, y = iris.data, iris.target
    inst = ArtificialNeuralNetwork(in_nodes_num=None,
                                   out_nodes_num=None,
                                   hidden_nodes_num=None,
                                   learning_rate=0.001,
                                   bias=True,
                                   hidden_activ="sigmoid",
                                   output_activ="sigmoid")
    inst.auto_para_base_on_train_data(X, y)
    inst.init_weights()
    shuffled_index = np.random.permutation(y.shape[0])
    shuffled_X = X[shuffled_index, :]
    shuffled_y = y[shuffled_index]
    
    ratio = 2 / 3
    segment = int(y.shape[0] * ratio)
    
    train_X = shuffled_X[:segment]
    train_y = shuffled_y[:segment]
    
    test_X = shuffled_X[segment:]
    test_y = shuffled_y[segment:]
    for i in range(9999):
        inst.train(train_X, train_y)
        _, output_label = inst.predict(train_X)  #输出预测结果  output predict labels
        accuracy = len(train_y[output_label == train_y]) / len(train_y)  #统计准确率  calc the accuracy
        print(accuracy)
        if accuracy == 1.:
            break
        
    print("Done!")

if __name__ == "__main__":
    main()
