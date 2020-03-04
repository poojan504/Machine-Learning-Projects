#######################################################################################################################
# Import some Libraries to use in the prediction of the bill whether it is counterfeit ot not
#######################################################################################################################
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from pandas import DataFrame
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import cm as cm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


#######################################################################################################################
# load the data  and correlation matrix to find the correlation between the co efficients
#######################################################################################################################

def Data():
    url = "data_banknote_authentication.txt"
    names = ["variance of Wavelet Transformed image", "skewness of Wavelet Transformed image",
             "curtosis of Wavelet Transformed image", "entropy of image", "class"]
    data = pd.read_csv(url, names=names)
    dataframe = DataFrame(data)

    data.head()
    print(data.groupby('class').size())
    print(data.describe())

    # to store the data into features and the targets
    features = dataframe.iloc[:, 0:4]
    Targets = dataframe.iloc[:, -1]

    return features, Targets, dataframe


#######################################################################################################################
# Created the function for the each algorithm
#######################################################################################################################

def Logistic(features, Targets):
    # split the data into train and test  variables to train and predict the outcome
    x_train, x_test, y_train, y_test = model_selection.train_test_split(features, Targets, test_size=0.3,
                                                                        random_state=0)
    # Standardize the data
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)  # apply to the training data
    x_test_std = sc.transform(x_test)
    # training the model
    model = LogisticRegression()
    model.fit(x_train_std, y_train)
    # predicting the output
    y_pred = model.predict(x_test_std)

    misclass_samples = (y_test != y_pred).sum()
    accuracy = accuracy_score(y_test, y_pred)
    print("######################Logistic#########################")
    print('Misclassified samples: %d' % misclass_samples)  # how'd we do?
    print('Accuracy: %.2f' % accuracy)
    return accuracy


def Percp(features, Targets):
    # split the data into train and test  variables to train and predict the outcome
    x_train, x_test, y_train, y_test = model_selection.train_test_split(features, Targets, test_size=0.3,
                                                                        random_state=0)
    # Standardize the data
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)  # apply to the training data
    x_test_std = sc.transform(x_test)
    # training the model
    ppn = Perceptron()
    ppn.fit(x_train_std, y_train)
    # predicting the  value
    y_pred = ppn.predict(x_test_std)

    misclass_samples = (y_test != y_pred).sum()
    accuracy = accuracy_score(y_test, y_pred)
    print("######################Perceptron#########################")
    print('Misclassified samples: %d' % misclass_samples)  # how'd we do?
    print('Accuracy: %.2f' % accuracy)
    return accuracy


def SVM(features, Targets):
    # split the data into train and test  variables to train and predict the outcome
    x_train, x_test, y_train, y_test = model_selection.train_test_split(features, Targets, test_size=0.3,
                                                                        random_state=0)
    # Standarize the Data
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)  # apply to the training data
    x_test_std = sc.transform(x_test)
    # Train the model
    accuracy_vs_neighbors = []
    svm = SVC(kernel='rbf', C=1.0, random_state=0)
    svm.fit(x_train_std, y_train)  # do the training
    # Predict the Output
    y_pred = svm.predict(x_test_std)

    misclass_samples = (y_test != y_pred).sum()
    accuracy = accuracy_score(y_test, y_pred)
    print("######################Standard Vector Matrix#########################")
    print('Misclassified samples: %d' % misclass_samples)  # how'd we do?
    print('Accuracy: %.2f' % accuracy)
    return accuracy


def decision_tree(features, Targets):
    # split the data into train and test  variables to train and predict the outcome
    x_train, x_test, y_train, y_test = model_selection.train_test_split(features, Targets, test_size=0.3,
                                                                        random_state=0)
    # Standardize the Data
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)  # apply to the training data
    x_test_std = sc.transform(x_test)
    # training the model
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
    tree.fit(x_train_std, y_train)
    # Predict the output
    y_pred = tree.predict(x_test_std)

    misclass_samples = (y_test != y_pred).sum()
    accuracy = accuracy_score(y_test, y_pred)
    print("######################Decision Tree#########################")
    print('Misclassified samples: %d' % misclass_samples)  # how'd we do?
    print('Accuracy: %.2f' % accuracy)
    return accuracy


def Random(features, Targets):
    # split the data into train and test  variables to train and predict the outcome
    x_train, x_test, y_train, y_test = model_selection.train_test_split(features, Targets, test_size=0.3,
                                                                        random_state=0)
    # Standardize the data
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)  # apply to the training data
    x_test_std = sc.transform(x_test)
    # Train the model
    Rand = RandomForestClassifier()
    Rand.fit(x_train_std, y_train)
    # predict the output
    y_pred = Rand.predict(x_test_std)

    misclass_samples = (y_test != y_pred).sum()
    accuracy = accuracy_score(y_test, y_pred)
    print("######################Random Forest#########################")
    print('Misclassified samples: %d' % misclass_samples)  # how'd we do?
    print('Accuracy: %.2f' % accuracy)
    return accuracy


def KNN(features, Targets):
    # split the data into train and test  variables to train and predict the outcome
    x_train, x_test, y_train, y_test = model_selection.train_test_split(features, Targets, test_size=0.3,
                                                                        random_state=0)
    # Standardize the data
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)  # apply to the training data
    x_test_std = sc.transform(x_test)
    # Train the model
    Knn = KNeighborsClassifier()
    Knn.fit(x_train_std, y_train)
    # predict the output
    y_pred = Knn.predict(x_test_std)

    misclass_samples = (y_test != y_pred).sum()
    accuracy = accuracy_score(y_test, y_pred)
    print("######################KNN#########################")
    print('Misclassified samples: %d' % misclass_samples)  # how'd we do?
    print('Accuracy: %.2f' % accuracy)
    return accuracy


features, Target, df = Data()
LR = Logistic(features, Target)
P = Percp(features, Target)
SV = SVM(features, Target)
DT = decision_tree(features, Target)
RN = Random(features, Target)
KN = KNN(features, Target)
algo = [LR, P, SV, DT, RN, KN]
name = ["Logistic:", "Perceptron:", "SVM", "DT", "Random", "KNN"]
m = int(0)
# n = int
max_name = "sample"
print("#################################################################################\n Algorithm with highest "
      "accuracy is given "
      "by\n#################################################################################")
for i in range(0, len(algo)):
    if algo[i] >= m:

        accuracy = algo[i]
        m = algo[i]
        max_name = name[i]
        if i > 1:
            print(max_name, 'Accuracy: %.2f' % accuracy)
