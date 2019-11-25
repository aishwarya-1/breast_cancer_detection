#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math
from sklearn.model_selection import KFold
import statistics
import itertools

class KNN():
    def __init__(self, traindata=0, trainclass=0, testdata=0, testclass=0, optimal_k=5):

        self.X_train = traindata
        self.y_train = trainclass
        self.X_test = testdata
        self.y_test = testclass
        self.precision = 0
        self.recall = 0
        self.specificity = 0
        self.y_pred = []
        self.acc = 0
        
        self.k = optimal_k
        
        # Find the min and max values for each column
    def dataset_minmax(self, dataset):
        minmax = list()
        for i in range(len(dataset[0])):
            col_values = [row[i] for row in dataset]
            value_min = min(col_values)
            value_max = max(col_values)
            minmax.append([value_min, value_max])
        return minmax

    # Rescale dataset columns to the range 0-1
    def normalize_dataset(self):
        minmax = self.dataset_minmax(self.X_train)
        for row in self.X_train:
            for i in range(len(row)):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
                
        minmax = self.dataset_minmax(self.X_test)                
        for row in self.X_test:
            for i in range(len(row)):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
        
    def compute_confusion_mat(self):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(len(self.y_test)):
            if(self.y_test[i]==self.y_pred):
                if(self.y_test[i]==2):
                    tp += 1
                else:
                    tn += 1
            else:
                if(self.y_test[i]==2):
                    fp += 1
                else:
                    fn += 1
        
        return [tp, fp, tn, fn]
    
    def params(self):
        l = compute_confusion_mat()
        self.precision = l[0]/(l[0]+l[1])
        self.recall = l[0]/(l[0]+l[3])
        self.specificity = (l[2]) / (l[2] + l[1])
        
        
    def accuracy(self):
        correct = 0
        for i in range(len(self.y_test)):
            if(self.y_test[i]==self.y_pred[i]):
                correct = correct + 1
        return (correct/len(self.y_test))*100
    
    def euclidean_distance(self, point1, point2):
        sum_squared_distance = 0
        for i in range(len(point1)):
            sum_squared_distance += math.pow(point1[i] - point2[i], 2)
        return math.sqrt(sum_squared_distance)  
    
    def mode(self, labels):
        return Counter(labels).most_common(1)[0][0]
    
    def knn(self, query):
        neighbor_distances_and_indices = []

        for index, example in enumerate(self.X_train):
            distance = self.euclidean_distance(example, query)
            neighbor_distances_and_indices.append((distance, index))

        sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)

        k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:self.k]

        k_nearest_labels = [self.y_train[i] for distance, i in k_nearest_distances_and_indices]

        return self.mode(k_nearest_labels) 
    
    def knn_classifier(self):
        self.normalize_dataset()
        for i in self.X_test:
            clf_prediction = self.knn(i)
            self.y_pred.append(clf_prediction)
            
        self.acc = self.accuracy()

        return self.acc