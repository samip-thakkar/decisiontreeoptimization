# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 19:36:46 2019

@author: Samip
"""

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load the dataset
data = pd.read_csv('Social_Network_Ads.csv')

"""We will take only Age and Salary as independent attributes"""

#Prepare a feature matrix
features_cols = ['Age', 'EstimatedSalary']
X = data.iloc[:, [2,3]].values
y = data.iloc[:, -1].values

#Split the dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Perform feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#Fit the model in Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt = dt.fit(X_train, y_train)

#Make prediction 
y_pred = dt.predict(X_test)

#Calculate Accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print("Accuracy of Decision Tree: ", str(acc))

#Print Confusion Matrix
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_test, y_pred)
print(cf)

#Visualize the model prediction results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, dt.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(("red", "green")))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(("red", "green"))(i), label = j)
plt.title("Decision Tree Test set results")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

#Visualize a tree
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
dot_data = StringIO()
export_graphviz(dt, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = features_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_png('Before Optimization.png')


"""OPTIMIZING THE DECISION TREE"""

"""For optimization, criterion is set to 'gini' for Gini Index, splitter is set to best to choose the best split, max_depth is set to control
Overfitting and Underfitting. Higher value gives overfitting and lower value gives underfitting""" 

classifier = DecisionTreeClassifier(criterion = 'gini', max_depth = 3)
classifier = classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print('After Optimization Accuracy:', accuracy_score(y_test, y_pred))

#View Decision Tree again
dot_data = StringIO()
export_graphviz(classifier, out_file = dot_data,  
                filled = True, rounded = True,
                special_characters = True, feature_names = features_cols,class_names = ['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_png('After Optimization.png')