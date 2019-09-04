# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 13:27:45 2019

@author: Dong
"""

import numpy as np
import pickle

class ID3:
    
    def __init__(self,
                 features_name=None,
                 max_depth=None,
                 min_samples_split=2):
        self.features_name = features_name
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        
        
    def _check_input(self, arr):
        if list == type(arr):
            return np.array(arr)
        elif np.ndarray == type(arr):
            return arr
        return None
    
    
#    def _cnt_majority(self, class_arr):
#        class_cnt = {}
#        max_class_cnt = -1
#        for class_i in class_arr:
#            if class_i not in class_cnt.keys():
#                class_cnt[class_i] = 0
#            class_cnt[class_i] += 1
#            if class_cnt[class_i] > max_class_cnt:
#                max_class_cnt = class_cnt[class_i]
#                max_class_label = class_i
#                        
#        return max_class_label
    
        
    def _calc_shannon_entropy(self, y):
        label_cnt = {}
        for label in y:
            if label not in label_cnt.keys():
                label_cnt[label] = 0
            label_cnt[label] += 1
            
        shannon_entropy = 0.0
        for key in label_cnt:
            prob = label_cnt[key] / len(y)
            shannon_entropy -= prob*np.log2(prob)
            
        return shannon_entropy
        
    
    def _split_according_feature(self, X, y, feature_index):
        X_subset_dict = {}
        y_subset_dict = {}
        for i in range(len(X)):
            tmpLine = X[i]
            key = X[i][feature_index]
            
            if key not in X_subset_dict.keys():
                tmpArr = []
                tmpArr.append(tmpLine)
                X_subset_dict[key] = tmpArr
                y_subset_dict[key] = []
            else:
                X_subset_dict[key].append(tmpLine)
            
            y_subset_dict[key].append(y[i])

        return X_subset_dict, y_subset_dict


    def _choose_feature_to_split(self, X, y):
        feature_num = len(X[0])
        entropy_base = self._calc_shannon_entropy(y)
        best_info_gain = 0.0
        best_feature_index = -1
        
        for i in range(feature_num):
            _, y_subset_dict = self._split_according_feature(X, y, i)
            entropy_tmp = 0.0
            for key in y_subset_dict.keys():
                arr = y_subset_dict[key]
                prob = len(arr) / len(y)
                entropy_tmp += prob * self._calc_shannon_entropy(arr)
            info_gain = entropy_base - entropy_tmp
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature_index = i
                
        return best_feature_index             
    
            
    def fit(self, X, y):#非递归
        try: 
            X = self._check_input(X)
            y = self._check_input(y)
            root_feature_idx = self._choose_feature_to_split(X, y)
            node_stack = []  #节点栈，保存当前访问的节点
            features_name = None
            if self.features_name != None:
                features_name = self._check_input(self.features_name)
                root_node = {features_name[root_feature_idx]: {}}
                node_stack.append(root_node[features_name[root_feature_idx]])
            else:
                root_node = {root_feature_idx: {}}
                node_stack.append(root_node[root_feature_idx])
                
            #类标号栈，保存当前进行划分的特征索引
            feature_idx_stack = []
            feature_idx_stack.append(root_feature_idx)
            
            #数据栈，保序当前需要被划分的数据，和类标号y_stack一一对应
            X_stack = []
            X_stack.append(X)
            y_stack = []
            y_stack.append(y)
            
            while(len(X_stack) > 0):
                X = X_stack.pop()
                y = y_stack.pop()
                feature_idx = feature_idx_stack.pop()
                p_current = node_stack.pop()  #指向当前节点的指针
                
                X_subset_dict, y_subset_dict = self._split_according_feature(X, y, feature_idx)
                
                for key in y_subset_dict.keys():
                    X_arr = X_subset_dict[key]
                    y_arr = y_subset_dict[key]
                    #如果全部属于某一类，则标记其类别，代表已经划分完成
                    class_set = set(y_arr)
                    if 1 == len(class_set):
                        end_label = class_set.pop()
                        if features_name is not None:
                            feature_name = features_name[feature_idx]
                            #分情况讨论，当节点具有分支的时候
                            if feature_name in p_current.keys():
                                p_current[feature_name][key] = end_label
                            else:
                                p_current[key] = end_label
                        else:
                            if feature_idx in p_current.keys():
                                p_current[feature_idx][key] = end_label
                            else:
                                p_current[key] = end_label
                        
                        continue
                        
                    if(len(X_arr[0]) > 0):
                        feature_idx_tmp = self._choose_feature_to_split(
                                X_arr, y_arr)
                        if features_name is not None:
                            feature_name_tmp = features_name[feature_idx_tmp]
                            p_current[key] = {}
                            p_current[key][feature_name_tmp] = {}
                        else:
                            p_current[key] = {}
                            p_current[key][feature_idx_tmp] = {}
                            
                        X_stack.append(X_arr)
                        y_stack.append(y_arr)
                        feature_idx_stack.append(feature_idx_tmp)
                        node_stack.append(p_current[key])
                            
                        
            return root_node
            
        except Exception as e:
            print(e)
    
            
    def predict(self, built_tree, X):
        try: 
            X = self._check_input(X)
            features_name = self._check_input(self.features_name)
            btk = built_tree.keys()
            first_feature_key = list(btk)[0]
            first_feature_value = built_tree[first_feature_key]
            if features_name is not None:
                feature_index = features_name.tolist().index(first_feature_key)
            else:
                feature_index = first_feature_key
            class_label = []
            for dat in X:
                for key in first_feature_value.keys():
                    if dat[feature_index] == key:
                        if type(first_feature_value[key]).__name__ == 'dict':
                            class_label = self.predict(first_feature_value[key], X)
                        else:
                            class_label.append(first_feature_value[key])
                        
            return class_label
        except Exception as e:
            print(e)
                        
       
    def save_built_tree(self, file_name, built_tree):
        try:
            fw = open(file_name, 'wb')
            pickle.dump(built_tree, fw)
            fw.close()
        except Exception as e:
            print(e)
        
        
    def load_built_tree(self, file_name):
        try:
            return pickle.load(open(file_name, 'rb'))
        except Exception as e:
            print(e)
        


def main():
    X = [[1,1],[1,1],[1,0],[0,1],[0,1]]
    y = ['yes', 'yes', 'no', 'no', 'no']
    features_name = ['f1', 'f2']
    model = ID3(features_name=features_name)
    #model = ID3()
    print(model._calc_shannon_entropy(y))
    X_subset_dict, y_subset_dict = model._split_according_feature(X, y, 0)
    print("X_subset: ", X_subset_dict)
    print("y_subset: ", y_subset_dict)
    built_tree = model.fit(X, y)
    print(built_tree)
    print(model.predict(built_tree, [[1,0], [1,1]]))
    model.save_built_tree('built_tree.m', built_tree)
    print(model.load_built_tree('built_tree.m'))
    print(type(model.load_built_tree('built_tree.m')))


if __name__ == "__main__":
    main()