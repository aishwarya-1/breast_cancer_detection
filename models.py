"""
    DecisionTreeClassifier Usage:

    from models import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
"""
import pandas as pd
from numpy import log2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math
from sklearn.model_selection import KFold
import statistics
import itertools


class DecisionTreeClassifier:

    def __init__(self):
        self._tree = {}
        self._default = None


    def _entropy(self,target):
        """
        Calculate the entropy of a dataset.
        The only parameter of this function is the target parameter which specifies the target column (as a pandas Series object)
        """
        target_unique = target.unique()
        target_counts = target.value_counts()
        E_s = 0
        for val in target_unique:
            ratio = target_counts[val]/len(target)
            E_s += -ratio*log2(ratio)
        return E_s


    def _information_gain(self,df,attribute_name,target_name="class"):
        """
        Calculate the information gain of a dataset. This function takes three parameters:
        1. df = The dataset for whose feature the Information Gain should be calculated
        2. attribute_name = the name of the feature for which the information gain should be calculated
        3. target_name = the name of the target feature. The default value is "class"
        """
        E_s = self._entropy(df[target_name])
        attribute_unique = df[attribute_name].unique()
        attribute_counts = df[attribute_name].value_counts()
        size = len(df)

        E_sum = 0
        for attr in attribute_unique:
            attr_ratio = attribute_counts[attr]/size            # |Sv|/|S|
            temp = df[target_name][df[attribute_name] == attr]  # list of df[target_name] for attr
            E_Sv = self._entropy(temp)                                # entropy of attr (E_Sv)
            E_sum += attr_ratio*E_Sv
        return E_s - E_sum


    def _build_tree(self,data,original_data,features,target_name="class",parent_node_class=None):
        """
        ID3 Algorithm
        This function takes five paramters
        1. data = the data for which the ID3 algorithm should be run --> In the first run this equals the total dataset
    
        2. original_data = This is the original dataset needed to calculate the mode target feature value of the original dataset
        in the case the dataset delivered by the first parameter is empty

        3. features = the feature space of the dataset . This is needed for the recursive call since during the tree growing process
        we have to remove features from our dataset --> Splitting at each node

        4. target_name = the name of the target attribute

        5. parent_node_class = This is the value or class of the mode target feature value of the parent node for a specific node. This is 
        also needed for the recursive call since if the splitting leads to a situation that there are no more features left in the feature
        space, we want to return the mode target feature value of the direct parent node.
        """ 
        
        # if dataset empty, return mode of target from original_data
        if len(data) == 0:
            return original_data[target_name].mode() 

        # if all the values in the target column are same, return that value
        elif len(data[target_name].unique()) <= 1:
            return data[target_name].unique()[0]       

        # if features is empty, return the mode target feature value of the direct parent node
        elif len(features) == 0:
            return parent_node_class
        
        # grow the tree
        else:
            # set parent_node_class to mode of target column of the current node
            parent_node_class = data[target_name].mode()

            # select the feature which best splits the dataset
            # return the information gain values for the features in the dataset
            item_values = [self._information_gain(data,feature,target_name) for feature in features] 
            best_feature_index = item_values.index(max(item_values))
            best_feature = features[best_feature_index]

            # create the tree structure
            tree = {best_feature: {} }

            # remove best_feature from features
            features = features.delete(best_feature_index)

            # grow a branch under root for each possible value the root node can take

            for value in data[best_feature].unique():
                # split the dataset
                data_subset = data.where(data[best_feature] == value).dropna()

                # call build_tree() with new parameters
                tree_sub = self._build_tree(data_subset,data,features,target_name,parent_node_class)

                # attach tree_sub to main tree
                tree[best_feature][value] = tree_sub


            return tree


    def _predict_query(self,query, tree, default):
        for key in query:
            if key in tree:
                try:
                    result = tree[key][query[key]] 
                except:
                    return default
                
                result = tree[key][query[key]]
                
                if type(result) == dict:
                    return self._predict_query(query,result,default)
                else:
                    return result
    

    def fit(self, X_train, y_train):
        data = pd.concat([X_train,y_train],axis=1)
        self._tree = self._build_tree(data, data, X_train.columns, y_train.name)
        self._default = y_train.mode()[0]


    def predict(self,X_test):
        X_test_dicts = X_test.to_dict(orient='records')
        y_pred = []
        for query in X_test_dicts:
            val = self._predict_query(query,self._tree,self._default)
            y_pred.append(val)
        return pd.Series(y_pred).astype(int)

    def get_decision_tree(self):
        return self._tree



class KNN():
    def __init__(self, traindata=0, trainclass=0, testdata=0, testclass=0, optimal_k=7):

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
        for i in self.X_test:
            clf_prediction = self.knn(i)
            self.y_pred.append(clf_prediction)
            
        self.acc = self.accuracy()

        return self.acc