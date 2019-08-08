# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 13:27:45 2019

@author: Dong
"""

import numpy as np
import pickle

class ID3:
    
    def __init__(self):
        pass
        
    def _check_input(self, arr):
        if list == type(arr):
            return np.array(arr)
        elif np.ndarray == type(arr):
            return arr
        return None
    
    
    def _cnt_majority(self, class_arr):
        class_cnt = {}
        max_class_cnt = -1
        for class_i in class_arr:
            if class_i not in class_cnt.keys():
                class_cnt[class_i] = 0
            class_cnt[class_i] += 1
            if class_cnt[class_i] > max_class_cnt:
                max_class_cnt = class_cnt[class_i]
                max_class_label = class_i
                        
        return max_class_label
    
    
    def _calc_shannon_entropy(self, dat):
        label_cnt = {}
        for sample in dat:
            label_i = sample[-1]
            if label_i not in label_cnt.keys():
                label_cnt[label_i] = 0
            label_cnt[label_i] += 1
            
        shannon_entropy = 0.0
        for key in label_cnt:
            prob = label_cnt[key] / len(dat)
            shannon_entropy -= prob*np.log2(prob)
            
        return shannon_entropy


    def _split_according_feature(self, dat, feature_index, feature_value):
        flg = False
        for sample_arr in dat:
            if feature_value == sample_arr[feature_index]:
                reduced_arr = np.concatenate(
                        (sample_arr[:feature_index], 
                         sample_arr[feature_index+1:]), axis=0)
                reduced_arr = reduced_arr.reshape(1, -1)
                if False == flg:
                    dat_subset = reduced_arr
                    flg = True
                else:
                    dat_subset = np.concatenate(
                            (dat_subset, reduced_arr), axis=0)
                         
        return dat_subset    

    
    def _choose_feature_to_split(self, dat):
        feature_num = dat[0].size - 1
        entropy_base = self._calc_shannon_entropy(dat)
        best_info_gain = 0.0
        best_feature_index = -1
        for feature_i in range(feature_num):
            sample_arr = np.array([s[feature_i] for s in dat])
            unique_set = set(sample_arr)
            entropy_tmp = 0.0
            for val in unique_set:
                dat_subset = self._split_according_feature(dat, feature_i, val)
                prob = dat_subset.shape[0] / dat.shape[0]
                entropy_tmp += prob * self._calc_shannon_entropy(dat_subset)
            info_gain = entropy_base - entropy_tmp
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature_index = feature_i
                
        return best_feature_index            
    
    
    def create_tree(self, dat, features_name, 
                    max_depth=None, min_samples_split=2):
        try: 
            dat = self._check_input(dat)
            features_name = self._check_input(features_name)
            class_arr = np.array([sample[-1] for sample in dat])
            cnt_first = np.compress(class_arr[0] == class_arr, class_arr, axis=0)
            if cnt_first.size == class_arr.size:
                return class_arr[0]
            if 1 == dat[0].size or 0 == max_depth:
                return self._cnt_majority(class_arr)
            if dat.shape[0] < min_samples_split:
                return self._cnt_majority(class_arr)
            
            best_feature_index = self._choose_feature_to_split(dat)
            key_str = features_name[best_feature_index]
            tree_tmp = {key_str: {}}
            features_name = np.delete(features_name, best_feature_index)
            sample_arr = np.array([s[best_feature_index] for s in dat])
            unique_set = set(sample_arr)
            for val in unique_set:
                if None == max_depth:
                    tree_tmp[key_str][val] = self.create_tree(
                            self._split_according_feature(dat, best_feature_index, val),
                            features_name, None, min_samples_split)
                else:
                    tree_tmp[key_str][val] = self.create_tree(
                            self._split_according_feature(dat, best_feature_index, val),
                            features_name, max_depth - 1, min_samples_split)
                
            return tree_tmp
            
        except Exception as e:
            print(e)
    
    
    def predict(self, built_tree, dat, features_name):
        try: 
            dat = self._check_input(dat)
            features_name = self._check_input(features_name)
            btk = built_tree.keys()
            first_feature_key = list(btk)[0]
            first_feature_value = built_tree[first_feature_key]
            feature_index = features_name.tolist().index(first_feature_key)
            for key in first_feature_value.keys():
                if dat[feature_index] == int(key):
                    if type(first_feature_value[key]).__name__ == 'dict':
                        class_label = self.predict(first_feature_value[key],
                                                   dat, features_name)
                    else:
                        class_label = first_feature_value[key]
                        
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
    dat = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    features_name = ['f1', 'f2']
    model = ID3()
    print(model._calc_shannon_entropy(dat))
    print(model._split_according_feature(dat, 0, 1))
    built_tree = model.create_tree(dat, features_name)
    print(model.predict(built_tree, [1,0], features_name))
    model.save_built_tree('built_tree.m', built_tree)
    print(model.load_built_tree('built_tree.m'))


if __name__ == "__main__":
    main()