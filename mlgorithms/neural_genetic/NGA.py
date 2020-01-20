# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:57:50 2020

@author: yunchengdong
"""

from mlgorithms.neural_genetic.ANN import *
from pylab import mpl
from sklearn.datasets import load_iris

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pk
import time
import warnings
warnings.filterwarnings("ignore")  #为了在控制台看输出结果，忽略warning打印输出  ignore warnings in console

plt.style.use("seaborn-dark")  #绘图风格选择  Choose a drawing style
mpl.rcParams['font.sans-serif'] = ['SimHei']  #解决中文显示问题  solve Chinese cannot be displayed
mpl.rcParams['axes.unicode_minus'] = False  #解决坐标轴负号显示问题  Solve the axis minus sign cannot be displayed


class NeuralGeneticAlgorithm(ArtificialNeuralNetwork):
    
    def __init__(self, pop_size=20,  #种群大小  population size
                 gen_iter_num=100,  #遗传算法迭代次数  iterations of genetic algorithm  
                 p_crossover=0.8,  #交叉概率  crossover probability
                 p_mutation=0.05,  #变异概率  mutation probability
                 in_nodes_num=None,  #输入层神经元个数  number of neurons in input layer
                 hidden_nodes_num=10,  #隐藏层神经元个数  number of neurons in hidden layer
                 out_nodes_num=None,  #输出层神经元个数  number of neurons in output layer
                 learning_rate=0.01,  #学习率，用于更新权重  used to update weights
                 bias=True,  #偏置量  y=wx+b  b is the bias
                 activation_func="sigmoid",  #激活函数  activation function  
                 ann_iter_num=1000  #神经网络迭代次数  iterations of neural network
                 ):
        
        ArtificialNeuralNetwork.__init__(self, in_nodes_num=in_nodes_num,
                                         hidden_nodes_num=hidden_nodes_num,
                                         out_nodes_num=out_nodes_num,
                                         learning_rate=learning_rate,
                                         activation_func=activation_func,
                                         bias=bias
                                         )

        self.pop_size = pop_size
        self.gen_iter_num = gen_iter_num
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        #self.init_pop_weights = []  #遗传算法种群初始权重  Initial weights of population in genetic algorithm
        self.ann_iter_num = ann_iter_num
        
        self.pop_info = None  #种群信息，包括适应度、BP权重等  population infomation, include fitness value, BP weights...
        
        #self.path = os.path.split(os.path.realpath(__file__))[0]
    
    
    def evaluate(self):
        '''计算种群适应度  Calc population fitness
        适应度为准确率
        The fitness is the predict accuracy
        
        '''
        for _ in range(self.ann_iter_num):
            self.train(self.X, self.y)  #BP训练  train BP network
            _, output_label = self.predict(self.X)  #输出预测结果  output predict labels
            accuracy = len(self.y[output_label == self.y]) / len(self.y)  #统计准确率  calc the accuracy
            if accuracy == 1.:
                break
            
#        output_vec, output_label = self.predict(self.X)
#        err_abs = abs(output_vec - self.target_vec)
#        err_sum = np.sum(err_abs)
#        accuracy = len(self.y[output_label == self.y]) / len(self.y)
        fitness_val = accuracy
        
        #当选择输出误差的倒数作为适应度函数时，为什么会出现适应度值变大而准确率反而下降？
        #When the reciprocal of the output error is chosen as the fitness function, 
        #why does the fitness become larger but the accuracy decrease?
        #fitness_val = 1 / (1 + err_sum)
        #fitness_val = 1 / err_sum
        
        
        return fitness_val
        
                
    def pop_init(self):
        '''种群初始化  population initialization
        
        种群list末尾保存适应度最大的个体，所以len(self.pop_info)==len(pop_size)+1
        '''
        self.pop_info = []  #存储种群信息  to store population info
        for i in range(self.pop_size): 
            self.create_weight_matrices()  #生成初始权重矩阵  create init weight mat
            dict_tmp = {}
            mat1 = self.weights_hidden_in.copy()
            mat2 = self.weights_hidden_out.copy()
            dict_tmp["gene"] = [mat1, mat2]
            fitness_val = self.evaluate()  #计算个体适应度 Calculate each individual fitness
            dict_tmp["fitness"] = fitness_val
            dict_tmp["weights_hidden_in"] = self.weights_hidden_in  #BP网络隐藏层输入权重  input weights for hidden layer
            dict_tmp["weights_hidden_out"] = self.weights_hidden_out  #BP网络隐藏层输出权重  output weights for hidden layer
            dict_tmp["r_fitness"] = 0.  #个体适应度占种群总适应度的比例  Proportion of individual fitness to total fitness
            dict_tmp["c_fitness"] = 0.  #累积概率  Cumulative probability
            #dict_tmp["gene"] = [self.weights_hidden_in, self.weights_hidden_out]
            self.pop_info.append(dict_tmp)
            if self.pop_size - 1 == i:  #追加一个dict，用于保存适应度最大的个体  append one more is to store the individual of max fitness
                self.pop_info.append(dict_tmp)
                
        for i in range(self.pop_size):
            if self.pop_info[-1]["fitness"] < self.pop_info[i]["fitness"]:
                self.pop_info[-1] = self.pop_info[i].copy()  #把最大适应度的个体赋值给最后一个元素 Assign the individual with the maximum fitness to the last element
        
    
    def selector(self):
        '''轮盘赌算法  roulette algorithm
        '''
        fitness_sum = 0.
        for i in range(self.pop_size):
            fitness_sum += self.pop_info[i]["fitness"]  #计算种群总适应度  calc total fitness of the population
            
        for i in range(self.pop_size):
            #计算每个个体适应度占总适应度的比例 calc proportion of each individual fitness to total fitness
            self.pop_info[i]["r_fitness"] = self.pop_info[i]["fitness"] / fitness_sum
            
        self.pop_info[0]["c_fitness"] = self.pop_info[0]["r_fitness"]
        for i in range(1, self.pop_size):
            #计算每个个体的累积概率  calc cumulative of each individual
            self.pop_info[i]["c_fitness"] = self.pop_info[i-1]["c_fitness"] + self.pop_info[i]["r_fitness"]
            
        new_pop = [None] * (self.pop_size + 1)
        for i in range(self.pop_size):
            p = np.random.uniform(0., 1.)
            if p < self.pop_info[0]["c_fitness"]:
                new_pop[i] = self.pop_info[0].copy()
            else:
                for j in range(self.pop_size):
                    if self.pop_info[j]["c_fitness"] <= p < self.pop_info[j+1]["c_fitness"]:
                        new_pop[i] = self.pop_info[j+1].copy()
                        
        for i in range(self.pop_size):
            self.pop_info[i] = new_pop[i]
            
            
    def xover2(self, pos1=None, pos2=None):
        '''双点交叉: 把两个个体相同两点之间的基因进行交换
        Two point crossover: Exchange genes between the same two points of two individuals
        
        Args:
            pos1: 染色体下标索引  the index of chromosome
            pos2: 另一条染色体下标索引  the index of the other chromosome
        '''
        
        #每个染色体都是一维的，所以要求出把权重矩阵展平后的长度
        #Each chromosome is one-dimensional, so the length after flattening the weight matrix is required
        self.gene_len = self.hidden_nodes_num*(self.in_nodes_num+1) + self.out_nodes_num*(self.hidden_nodes_num+1)
        index1 = np.random.randint(0, self.gene_len)
        index2 = np.random.randint(0, self.gene_len)
        
        #异或实现整形数据交换 XOR for integer data exchange
        if index1 > index2:
            index1 ^= index2
            index2 ^= index1
            index1 ^= index2
            
        #需要深拷贝  deep copy
        arr_tmp = self.pop_info[pos1]["gene"][index1:index2].copy()
        self.pop_info[pos1]["gene"][index1:index2] = self.pop_info[pos2]["gene"][index1:index2]
        self.pop_info[pos2]["gene"][index1:index2] = arr_tmp
        
    
    
    def crossover(self):
        '''双点交叉: 把两个个体相同两点之间的基因进行交换
        Two point crossover: Exchange genes between the same two points of two individuals
        '''
        one = None
        first = 0
        for i in range(self.pop_size):
            p = np.random.uniform(0., 1.)
            if p < self.p_crossover:
                first += 1
                if 0 == first % 2:
                    self.xover2(one, i)
                else:
                   one = i 
        
    
    def mat2arr(self):
        '''为了后面的交叉和变异操作，需要把权重矩阵展平
        For the following crossover and mutation operations, the weight matrix needs to be flattened
        '''
        for i in range(self.pop_size):
            weights_mat = self.pop_info[i]["gene"].copy()
            arr1 = weights_mat[0].flatten()
            arr2 = weights_mat[1].flatten()
            self.pop_info[i]["gene"] = np.concatenate((arr1, arr2))
            
            
    def arr2mat(self):
        '''
        交叉、变异后的基因（即神经网络权重）要带入神经网络进行计算，所以需要把基因reshape成矩阵
        After crossover and mutation, gene (i.e. neural network weight) 
        should be brought into neural network for calculation, so gene should be made into matrix
        '''
        for i in range(self.pop_size):
            weights_mat = []
            weights_arr = self.pop_info[i]["gene"].copy()
            segment_position = self.hidden_nodes_num * (self.in_nodes_num + 1)
            mat1 = weights_arr[:segment_position].reshape(self.hidden_nodes_num, self.in_nodes_num + 1)
            mat2 = weights_arr[segment_position:].reshape(self.out_nodes_num, self.hidden_nodes_num + 1)
            weights_mat.append(mat1)
            weights_mat.append(mat2)
            self.pop_info[i]["gene"] = weights_mat
    

    def mutate(self):
        '''变异策略：在随机位置生成上下界之间的一个随机数
        Mutation strategy: generating a random number between upper and lower bounds of random position
        '''
        for i in range(self.pop_size):
            for j in range(self.gene_len):
                p = np.random.uniform(0, 1)
                if p < self.p_mutation:
                    if j < self.hidden_nodes_num * (self.in_nodes_num + 1):
                        self.pop_info[i]["gene"][j] = np.random.uniform(-self.rad1, self.rad1)
                    else:
                        self.pop_info[i]["gene"][j] = np.random.uniform(-self.rad2, self.rad2)
            
    
    def elitist(self):
        '''精英策略： 找到适应度最低和最高的个体，然后和种群末尾个体进行比较，
        假如适应度最高个体大于末尾个体，则把适应度最高个体赋值给末尾，否则把
        末尾个体赋值给适应度最低个体，这样做是为了保证最优个体能够遗传下去
        Elite strategy: find the individual with the lowest and highest fitness,
        and then compare it with the individual at the end of the population. 
        If the individual with the highest fitness is larger than the individual 
        at the end, assign the individual with the highest fitness to the end,
        otherwise assign the individual at the end to the individual with the 
        lowest fitness, so as to ensure that the optimal individual can inherit
        '''
        best_mem = -1
        worst_mem = -1
        best = self.pop_info[0]["fitness"]
        worst = self.pop_info[0]["fitness"]
        
        for i in range(self.pop_size - 1):
            if self.pop_info[i+1]["fitness"] < self.pop_info[i]["fitness"]:
                if best <= self.pop_info[i]["fitness"]:
                    best = self.pop_info[i]["fitness"]
                    best_mem = i
                if self.pop_info[i+1]["fitness"] <= worst:
                    worst = self.pop_info[i+1]["fitness"]
                    worst_mem = i + 1
            else:
                if self.pop_info[i]["fitness"] <= worst:
                    worst = self.pop_info[i]["fitness"]
                    worst_mem = i
                if best <= self.pop_info[i+1]["fitness"]:
                    best = self.pop_info[i+1]["fitness"]
                    best_mem = i + 1
                    
        if self.pop_info[self.pop_size]["fitness"] < best:
            self.pop_info[self.pop_size] = self.pop_info[best_mem].copy()
        else:
            self.pop_info[worst_mem] = self.pop_info[self.pop_size].copy()
             
    
    def model_fit(self, X=None, y=None):
        '''遗传神经网络  neural-genetic-algorithm
        
        Args:
            X: 特征数据  feature data
               ndarray or list, shape:(n_samples, n_features)
            y: 标签数据  label
               ndarray or list, shape:(n_samples,)
        '''
        try:
            self.X = np.array(X)
            self.y = np.array(y)
            
            if len(self.X.shape) != 2:
                raise ValueError("Feature data must be 2D!")
                
            if len(self.y.shape) != 1:
                raise ValueError("Label data must be one-dimensional!")
            
            self.pop_init()  #种群初始化  population initialization
            
            self.fitness_arr = np.ones(self.gen_iter_num)  #保存每代适应度最大值  Save the maximum fitness of each generation
            
            for i in range(self.gen_iter_num):
                self.selector()  #选择遗传到下一代的个体  Select individuals to pass on to the next generation
                
                #为了后面的交叉和变异操作，需要把权重矩阵展平
                #For the following crossover and mutation operations, the weight matrix needs to be flattened
                self.mat2arr()  
                
                self.crossover()  #交叉  crossover
                
                self.mutate()  #变异  mutation
                
                #交叉、变异后的基因（即神经网络权重）要带入神经网络进行计算，所以需要把基因reshape成矩阵
                #After crossover and mutation, gene (i.e. neural network weight) 
                #should be brought into neural network for calculation, so gene should be made into matrix
                self.arr2mat()
                
                for j in range(self.pop_size):
                    self.weights_hidden_in = self.pop_info[j]["gene"][0].copy()
                    self.weights_hidden_out = self.pop_info[j]["gene"][1].copy()
                    self.pop_info[j]["fitness"] = self.evaluate()  #计算适应度  calc the fitness
                    
                    #保存模型训练好的权重  Save the trained weight of the model
                    self.pop_info[j]["weights_hidden_in"] = self.weights_hidden_in
                    self.pop_info[j]["weights_hidden_out"] = self.weights_hidden_out
                    
                      
                self.elitist()  #精英策略  Elite strategy
                
                print("Generation ", i+1, ":")
                #保存每代适应度最大值  Save the maximum fitness of each generation
                self.fitness_arr[i] = self.pop_info[self.pop_size]["fitness"]
                print("Best fitness: ", self.fitness_arr[i])
                if self.fitness_arr[i] == 1.:
                    break
                 
                print("-------------------------------------")
                
            print("Done!")
            
        except Exception as e:
            print(e)
            
            
    def model_save(self, model_path=None):
        '''保存训练好的模型，其实也就是保存权重
        Save the trained model, in fact, save the weights
        
        Args:
            model_path: 模型保存路径，包含模型名字，例如：model_path="./model/test.model"
                        Model saving path, including model name, for example: model_path="./model/test.model"
                        
                        也可以为None，当为None则保存在“./model”路径下，且模型名字为："NGA_时间戳.model"
                        if model_path is None, model_path="./model/NGA_timestamp.model"
        '''
        
        try:
            best = self.pop_info[self.pop_size]
            if model_path is None:
                dir_name = "./model/"   #可能会报错
                is_exists=os.path.exists(dir_name)
                if not is_exists:
                    os.makedirs(dir_name)
                    
                model_path = dir_name + "NGA_" + time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())) + '.model'
                
            else:
                dir_name = model_path[:model_path.rfind("/")]
                is_exists=os.path.exists(dir_name)
                if not is_exists:
                    os.makedirs(dir_name)
            pk.dump((best["weights_hidden_in"], best["weights_hidden_out"]), open(model_path,'wb'))
        except Exception as e:
            print(e)
            
    
    def model_load(self, model_path=None):
        '''模型加载：用于加载保存好的模型
        Model loading: used to load a saved model
        '''
        try:
            if model_path is None:
                raise ValueError("Model path is None!")
            else:
                try:
                    model = pk.load(open(model_path, 'rb')) 
                except Exception as e:
                    print(e)
                    
                return model
                    
        except Exception as e:
            print(e)
        
            
    def model_predict(self, X=None, model=None):
        '''模型预测：可以在模型训练好后直接预测，也可以通过加载模型的方式进行预测
        Model prediction: it can be predicted directly after the model is trained, or it can be predicted by loading the model
        
        Args:
            X: 特征数据  feature data
               ndarray or list, shape:(n_samples, n_features)
            model: 加载的模型或None，当为None时表示对训练好的模型直接进行预测
                   When it is none, the trained model is predicted directly
                   
        '''
        try:
            X = np.array(X)
            if len(X.shape) != 2:
                raise ValueError("Feature data must be 2D!")
            
            if model is None:
                best = self.pop_info[self.pop_size]
                self.weights_hidden_in = best["weights_hidden_in"]
                self.weights_hidden_out = best["weights_hidden_out"]
            else:
                self.weights_hidden_in = model[0]
                self.weights_hidden_out = model[1]
                
            _, output_label = self.predict(X)
                
            return output_label
        
        except Exception as e:
            print(e)
            
            
    def fitness_plot(self):
        '''对每代的最大适应度进行绘图
        plot the maximum fitness of each generation
        '''
        
        try:
            plt.plot(range(self.gen_iter_num), self.fitness_arr)
            plt.xlabel("种群迭代次数")
            plt.ylabel("适应度")
            plt.grid()
            
            dir_name = "./images/"
            is_exists=os.path.exists(dir_name)
            if not is_exists:
                os.makedirs(dir_name)
                    
            fig_path = dir_name + "Fitness_" + time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())) + '.png'
            plt.savefig(fig_path, dpi=300)
            
            plt.show()
        except Exception as e:
            print(e)
        
        
        
###############################################################################
def main():

    #导入鸢尾花数据  import iris data
    iris = load_iris()
    X, y = iris.data, iris.target
    in_num = X.shape[1]
    out_num = np.unique(y).shape[0]
    inst = NeuralGeneticAlgorithm(in_nodes_num=in_num, out_nodes_num=out_num, #这两个为必须参数
                                  ann_iter_num=100, gen_iter_num=100, 
                                  pop_size=20, p_mutation=0.1)
    
    times = 10
    train_acc = np.zeros(times)
    test_acc = np.zeros(times)
    for i in range(times):
        shuffled_index = np.random.permutation(y.shape[0])
        shuffled_X = X[shuffled_index, :]
        shuffled_y = y[shuffled_index]
        
        ratio = 2 / 3
        segment = int(y.shape[0] * ratio)
        
        train_X = shuffled_X[:segment]
        train_y = shuffled_y[:segment]
        
        test_X = shuffled_X[segment:]
        test_y = shuffled_y[segment:]
        
        inst.model_fit(X=train_X, y=train_y)
        
        inst.fitness_plot()
        
        #inst.model_save("./model/test.model")
        
        #测试模型加载
#        model = inst.model_load("./model/test.model")
#        output_label = inst.model_predict(X=test_X, model=model)
#        test_accuracy = 0.
#        test_accuracy = len(test_y[output_label == test_y]) / len(test_y)
        
        
        pop_size = inst.pop_size
        best = inst.pop_info[pop_size]
        
        print("Train accuracy: ", best["fitness"])
        train_acc[i] = best["fitness"]
        
        output_label = inst.model_predict(X=test_X)
        
        test_accuracy = len(test_y[output_label == test_y]) / len(test_y)
        
        print("Test accuracy: ", test_accuracy)
        test_acc[i] = test_accuracy
        print("\n\n")
        
    print("Average train accuracy: ", np.mean(train_acc))
    print("Average test accuracy: ", np.mean(test_acc))
    
    dir_name = "./accuracy/"   #可能会报错
    is_exists=os.path.exists(dir_name)
    if not is_exists:
        os.makedirs(dir_name)
    name1 = dir_name + "train_acc_" + time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())) + ".para"
    name2 = dir_name + "test_acc_" + time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())) + ".para"
    pk.dump(train_acc, open(name1, "wb"))
    pk.dump(test_acc, open(name2, "wb"))
    


if __name__ == "__main__":
    main()
        
        
        
        
        
        
    