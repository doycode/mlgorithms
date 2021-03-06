# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:35:44 2019

@author: Dong
"""

import numpy as np

class NaiveBayes:
    
    def __init__(self):
        self.prob_dict = {}
        self.words_prob_dict = {}
    
    def _check_input(self, arr): 
        if list == type(arr):
            return np.array(arr)
        elif np.ndarray == type(arr):
            return arr
        return None
    
    
    def fit(self, sentence_vecs, class_label):
        try:
            sentence_vecs = self._check_input(sentence_vecs)
            class_label = self._check_input(class_label)
            vecs_num = sentence_vecs.shape[0]  #训练样本个数
            words_num = sentence_vecs[0].shape[0]  #特征个数
            #prob_dict = {}  #用于统计各类别出现概率
            vecs_sum_dict = {}  #用于各类别样本向量求和
            denom_dict = {}  #分母，用于求各特征属于某类的概率
            class_unique = np.unique(class_label)  #不重复的类别数组
            class_num = class_unique.shape[0]  #类别个数
            for i in range(class_num):
                class_name = class_unique[i]
                self.prob_dict[class_name] = np.argwhere(
                        class_name == class_label).shape[0] / vecs_num
                vecs_sum_dict[class_name] = np.ones(words_num)
                denom_dict[class_name] = 2.0
            
            for i in range(vecs_num):
                class_name = class_label[i]
                np.add(vecs_sum_dict[class_name], 
                       sentence_vecs[i], out=vecs_sum_dict[class_name])
                denom_dict[class_name] += np.sum(sentence_vecs[i])
                
            #words_prob_dict = {}  #各特征属于某类的概率
            for i in range(class_num):
                class_name = class_unique[i]
                self.words_prob_dict[class_name] = np.log(
                        vecs_sum_dict[class_name] / denom_dict[class_name])
            
            #return self.prob_dict, self.words_prob_dict
        
        except Exception as e:
            print(e)
                
            
    def predict(self, to_be_classified):
        try:
            to_be_classified = self._check_input(to_be_classified)
            class_label = []
            for sample in to_be_classified:
                max_prob = float("-inf")
                for key in self.prob_dict.keys():
                    prob_tmp = np.sum(sample * self.words_prob_dict[key]) + np.log(
                            self.prob_dict[key])
                    if(prob_tmp > max_prob):
                        max_prob = prob_tmp
                        max_key = key
                class_label.append(max_key)
            return class_label
        
        except Exception as e:
            print(e)
    
###############################################################################    
    
            
def create_words_set(dat):
    words_set = set([])
    for sentence in dat:
        words_set |= set(sentence)
        
    return list(words_set)


def sentences2vector(words_set, sentences):
    m = len(sentences)
    n = len(words_set)
    sentence_vecs = np.zeros(shape=(m, n))
    for i in range(len(sentences)):
        for word in sentences[i]:
            if word in words_set:
                idx = words_set.index(word)
                sentence_vecs[i][idx] += 1
            else:
                pass
            
    return sentence_vecs    
    
    
def main():
    #Test1
#    dat = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
#           ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
#           ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
#           ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
#           ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
#           ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
#    class_label = [0, 1, 0, 1, 0, 1]
#    
#    model = NaiveBayes()
#    words_set = create_words_set(dat)
#    print((words_set))
#    sentence_vecs = sentences2vector(words_set, dat)
#        
#    model.fit(sentence_vecs, class_label)
#    print(model.predict(sentence_vecs)) 
     
    #Test2
#    X = [[1,1],[1,1],[1,0],[0,1],[0,1]]
#    y = ['yes', 'yes', 'no', 'no', 'no']
#    model = NaiveBayes()
#    model.fit(X,y)
#    print(model.predict(X))
    
    #Test3
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    model = NaiveBayes()
    model.fit(X, y)
    y_pred = model.predict(X)
    print("Number of mislabeled points out of a total %d points : %d"
      % (iris.data.shape[0],(iris.target != y_pred).sum()))
    

if __name__ == '__main__':
    main()