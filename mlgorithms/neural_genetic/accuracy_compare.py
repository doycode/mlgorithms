# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 10:39:38 2020

@author: yunchengdong
"""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def main():
    # 读取数据  load data
    iris = load_iris()
    X, y = iris.data, iris.target

    # 模型字典  model dict
    model_dict = {'KNN': KNeighborsClassifier(n_neighbors=5), # KNN
                  'Logistic Regression': LogisticRegression(C=100), # 逻辑回归  logistic regression
                  'SVM': SVC(C=100)} # 支持向量机  support vector machine
    model_accuracy = {'KNN': 0., 'Logistic Regression': 0., 'SVM': 0.}
    times = 10  # 对各模型拟合10次，求在测试集上的平均准确率  to calc the average accuracy
    for i in range(times):
        # 分割数据集  split data set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3)
        print("Times", i+1, ":")
        for model_name, model in model_dict.items():
            # 训练模型  fit model
            model.fit(X_train,y_train)
            # 验证模型  Validate model on test set
            accuracy = model.score(X_test, y_test)
            model_accuracy[model_name] += accuracy
            print('{}模型的预测准确率为{:.2f}%'.format(model_name, accuracy*100))
        print("------------------------------------------------")
        
    for model_name in model_accuracy.keys():
        model_accuracy[model_name] /= times
        
    print(model_accuracy)

if __name__ == '__main__':
    main()
