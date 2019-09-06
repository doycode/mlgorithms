# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:06:43 2019

@author: Dong
"""

import numpy as np

class KNNClassifier:
    
    def __init__(self, X, y):
        self.X = self._check_input(X)
        self.y = self._check_input(y)

        
    def _check_input(self, arr):
        if list == type(arr):
            return np.array(arr)
        elif np.ndarray == type(arr):
            return arr
        return None        

    
    def _data_normalization(self, X):
        try:
            X = self._check_input(X)
            col_min = X.min(0)
            ranges = np.subtract(X.max(0), col_min)
            np.subtract(X, col_min, out=X)
            X = np.divide(X, ranges)
            
            return X, col_min, ranges
        except Exception as e:
            print(e)        

            
    def predict(self, test, k):
        
        try:
            test = self._check_input(test)
            if k < 1:
                raise ValueError("k must be a positive integer!")
            else:
                k = int(k)
            
            X, col_min, ranges = self._data_normalization(self.X)
            norm_test = (test - col_min) / ranges
            class_label = []
            for sample in norm_test:
                mat_square = (X - sample) ** 2  #广播机制
                distances = np.sqrt(mat_square.sum(axis=1))
                sorted_dis_index = distances.argsort()
                class_cnt = {}
                max_class_cnt = -1
                for i in range(k):
                    vote_label = self.y[sorted_dis_index[i]]
                    if vote_label not in class_cnt.keys():
                        class_cnt[vote_label] = 0
                    class_cnt[vote_label] += 1
                    if class_cnt[vote_label] > max_class_cnt:
                        max_class_cnt = class_cnt[vote_label]
                        max_class_label = vote_label
                        
                class_label.append(max_class_label)
            
            return class_label
            
        except Exception as e:
            print(e)
            
            
def main():
    X = [[1.0, 1.1], [1.0, 1.1], [0, 0], [0, 0.1]]
    y = ['A', 'A', 'B', 'B']
    model = KNNClassifier(X, y)
    print(model.predict([[0, 0], [1, 1]], 3))

if __name__ == "__main__":
    main()
            
            
        