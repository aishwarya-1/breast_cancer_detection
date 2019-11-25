import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math
from sklearn.model_selection import KFold
import statistics
import itertools

from numpy import log2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA

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

"""
    DecisionTreeClassifier Usage:

    from models import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
"""
# import pandas as pd
# from numpy import log2
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

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
        for key in list(query.keys()):
            if key in list(tree.keys()):
                try:
                    result = tree[key][query[key]] 
                except:
                    return default
                
                result = tree[key][query[key]]
                
                if isinstance(result,dict):
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

# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.model_selection import KFold, cross_validate, cross_val_score, train_test_split


class SVM():
    def __init__(self, dpstraindata=0, dpstrainclass=0, dpstestdata=0, dpstestclass=0, weights=0, confusion=2, learning_rate=1, predicted=0, parameters=0, entiredata=0, entireclass=0):
        self.confusionmat = [[0 for x in range(confusion)] for y in range(confusion)]
        self.confusionmatset = 0
        self.traindata = dpstraindata
        self.trainclass = dpstrainclass
        self.testdata = dpstestdata
        self.testclass = dpstestclass
        self.learning_rate = learning_rate
        self.predicted = predicted
        self.parameters = parameters
        self.weights = weights
        self.predictedsign = 0
        self.entiredata = entiredata
        self.entireclass = entireclass
        
    
    def train(self, dpsdataparam=0, dpsclassparam=0, epochs=10000):
        try:
            if(dpsdataparam == 0):
                pass
            else:
                self.traindata = dpsdataparam
        except:
            self.traindata = dpsdataparam
            
        try:
            if(dpsclassparam == 0):
                pass
            else:
                self.trainclass = dpsclassparam
        except:
            self.trainclass = dpsclassparam
            
        self.weights = np.zeros(3)
        
        learning_rate = 1
        
        for epoch in range(epochs):
            
            counter = 0
            for i, j, k in self.traindata:
                dotprod = np.dot(np.array([i,j, k]), self.weights)
                
                if(self.trainclass[counter] * dotprod < 1):
                    self.weights = self.weights + learning_rate * ((self.trainclass[counter] * self.traindata[counter]) - (2 * (1/epochs) * self.weights))
                    
                else:
                    self.weights = self.weights + learning_rate * (-2 * (1/epochs) * self.weights)
                    
                counter += 1

        return self.weights

    def predict(self, dpsdataparam=0, w=0):
        try:
            if(dpsdataparam == 0):
                pass
            else:
                self.testdata = dpsdataparam
        except:
            self.testdata = dpsdataparam
            
        weights = w
        
        try:
            if(w == 0):
                print("Precomputed weights used")
                weights = self.weights
                
        except:
            pass
        
        self.pred = list()
        
        for i, j, k in self.testdata:
            self.pred.append(np.dot(np.array([i, j, k]), weights))
        
        return self.pred
    
    def computeconfusionmat(self, dpspredicted=0, dpsobserved=0):
        self.predictedsign = np.sign(dpspredicted)
        
        try:
            if(dpsobserved == 0):
                pass
            else:
                self.testclass = dpsobserved
        except:
            self.testclass = dpsobserved
        
        for i in range(len(self.predicted)):
            if(self.predictedsign[i] == self.testclass[i]):
                if(self.testclass[i] == 1):
                    self.confusionmat[1][1] += 1
                else:
                    self.confusionmat[0][0] += 1
            else:
                if(self.testclass[i] == 1):
                    self.confusionmat[1][0] += 1
                else:
                    self.confusionmat[0][1] += 1
        self.confusionmatset = 1
        return self.confusionmat

    def computeparameters(self):
        if(self.confusionmatset != 1):
            print("Please compute the Confusion Matrix")
            return -1
        
        TP = self.confusionmat[1][1]
        TN = self.confusionmat[0][0]
        FP = self.confusionmat[0][1]
        FN = self.confusionmat[1][0]

        self.parameters = [0 for i in range(4)]
        # 1. Accuracy
        # 2. Precision
        # 3. Recall
        # 4. Specificity
        self.parameters[0] = (TP + TN)/(TP + TN + FP + FN)
        self.parameters[1] = (TP) / (TP + FP)
        self.parameters[2] = (TP) / (TP + FN)
        self.parameters[3] = (TN) / (TN + FP)

        return self.parameters

    def printparams(self):    
        self.parameters = self.computeparameters()
        
        try:
            if(self.parameters == -1):
                return -1
        except:
            pass
        
        print("Accuracy\t : - ",self.parameters[0])
        print("Precision\t : - ",self.parameters[1])
        print("Recall\t\t : - ",self.parameters[2])
        print("Specificity\t : - ",self.parameters[3])
    
    def accuracy(self, original=0, predicted=0, testdata=0):
        try:
            if(original == 0):
                pass
        
        except:
            self.testclass = original
        
        try:
            if(predicted == 0):
                pass
            else:
                self.predicted = predicted
        except:
            self.predicted = predicted
            
        
        try:
            if(testdata == 0):
                pass
        
        except:
            self.testdata = testdata
        
        try:
            if(self.predicted == 0):
                pass
        
        except:
            self.predicted = self.predict(self.testdata)
            
        signedop = np.sign(self.predicted)
        correct = np.sum(signedop == self.testclass)
        return ((correct/len(signedop)*100))
    
    def KFOLDaccuracy(self, splits=6, data=0, classip=0):
        try:
            if(data != 0):
                self.entiredata = data
            else:
                pass
        except:
            self.entiredata = data
            
        try:
            if(data != 0):
                self.entireclass = classip
            else:
                pass
        except:
            self.entireclass = classip
            
        kfold = KFold(n_splits=splits, random_state=None, shuffle=False)
        accuracies = list()
        
        pltnumber = 1
        for tr_ind, te_ind in kfold.split(self.entiredata, self.entireclass):
            self.traindata = self.entiredata[tr_ind]
            self.testdata = self.entiredata[te_ind]
            self.trainclass = self.entireclass[tr_ind]
            self.testclass = self.entireclass[te_ind]
            
            self.weights = self.train(self.traindata, self.trainclass, epochs=10000)
            self.predicted = self.predict(self.testdata, self.weights)
            accuracies.append(self.accuracy(self.testclass, self.predicted))
            
            self.plothyperplane(self.testdata, self.testclass, pltnumber=pltnumber, weights=self.weights)
            pltnumber += 1
            
        accuracies = np.array(accuracies)
        meanacc = np.mean(accuracies)
        return meanacc
    
    def plothyperplane(self, testdata=0, testclass=0, pltnumber=1, weights=0):
        try:
            if(testdata != 0):
                self.testdata = testdata
            else:
                pass
        except:
            self.testdata = testdata
        
        try:
            if(testclass != 0):
                self.testclass = testclass
            else:
                pass
        except:
            self.testclass = testclass
        
        try:
            if(weights != 0):
                self.weights = weights
            else:
                pass
        except:
            self.weights = weights
            
        plt.figure(pltnumber)
        
        counter = 0
        for i, j, k in self.testdata:
            if(self.testclass[counter] == 1):
                plt.scatter(i, j, s=120, marker='+', linewidths=2, color='red')
            else:
                plt.scatter(i, j, s=120, marker='_', linewidths=2, color='green')
            counter += 1

        a = -self.weights[0] / self.weights[1]
        xx = np.linspace(-2,3)
        yy = a * xx - (self.weights[2] / self.weights[1])
        plt.plot(xx, yy, 'k--')
        plt.show()

# KNN + DT
df  = pd.read_csv("cleaned_dataset.csv")
data = df.drop(["id","class"],axis=1)
target = df["class"]
X_train, X_test, y_train, y_test = train_test_split(data,target, train_size = 0.3, random_state=42)

# DT
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)
print("The accuracy of Decision Tree Classifier:",accuracy_score(y_test,y_pred_dt))

# KNN
knnobj = KNN(X_train.values.tolist(), y_train.values.tolist(), X_test.values.tolist(), y_test.values.tolist())
print("The accuracy of KNN classifier is ",knnobj.knn_classifier())

kf = KFold(n_splits=6,shuffle=True)
kf.get_n_splits(data)


accuracy_all = {
    'Decision_Tree':[],
    'k-NN': [],
    # 'SVM':[]
}
for train_index, test_index in kf.split(data):
    X_train, X_test = data.iloc[train_index], data.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]

    model_dt = DecisionTreeClassifier()
    model_dt.fit(X_train, y_train)
    y_pred_dt = model_dt.predict(X_test)
    y_acc_dt = accuracy_score(y_test,y_pred_dt)
    accuracy_all['Decision_Tree'].append(y_acc_dt*100)

    # model_knn
    knnobj = KNN(X_train.values.tolist(), y_train.values.tolist(), X_test.values.tolist(), y_test.values.tolist())
    y_acc_knn = knnobj.knn_classifier()
    y_pred_knn = knnobj.y_pred
    accuracy_all['k-NN'].append(y_acc_knn)

    # model_svm = 
    # model_svm.fit(X_train, y_train)
    # y_pred_svm = model_svm.predict(X_test)
    # y_acc_svm = accuracy_score(y_test,y_pred_svm)
    # accuracy_all['SVM'].append(y_acc_svm)

# Plot
df_temp = pd.DataFrame(accuracy_all)
df_temp = df_temp.reset_index()
df_temp.plot(x='index', y=['Decision_Tree','k-NN'])

# SVM 
dpsdf1 = pd.read_csv("cleaned_dataset.csv")
dpsdf1 = dpsdf1.drop("id",1)
dpsdf1data = dpsdf1[dpsdf1.columns[:-1]]
dpsdf1class = dpsdf1[dpsdf1.columns[-1]]

pca = PCA(n_components=2, whiten=True).fit(dpsdf1data)
dpsdf1data = pca.transform(dpsdf1data)
print('Preserved Variance: ', sum(pca.explained_variance_ratio_))

npdpsdf1data = np.array(dpsdf1data)
adddatabias = np.zeros((int(npdpsdf1data.shape[0]),1))
adddatabias.fill(-1)
npdpsdf1data = (np.append(npdpsdf1data, adddatabias, axis=1))
npdpsdf1class = np.array(dpsdf1class)
flag = 0

if(flag == 0):
    npdpsdf1class[npdpsdf1class < 3] = -1
    npdpsdf1class[npdpsdf1class > 3] = 1
    flag = 1

dpsdata_train, dpsdata_test, dpsclass_train, dpsclass_test = train_test_split(npdpsdf1data, npdpsdf1class, test_size= .4,random_state=0)
svmobj = SVM()
weights = svmobj.train(dpsdata_train, dpsclass_train, epochs=50)
print(weights)

predicted = svmobj.predict(dpsdata_test, weights)

svmaccuracy = svmobj.accuracy(dpsclass_test, predicted)
print(svmaccuracy)

svmobj.computeconfusionmat(predicted, dpsclass_test)

svmobj.printparams()

svmobj.KFOLDaccuracy(6, npdpsdf1data, npdpsdf1class)

svmobj.plothyperplane(dpsdata_test, dpsclass_test, weights=weights)


