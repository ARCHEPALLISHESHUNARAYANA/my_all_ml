# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'C:\Users\A.SHESHUNARAYANA\Desktop\iris.csv')
X = dataset.iloc[:, [2, 3]]
y = dataset.iloc[:, -1]
print(dataset.info())

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn import tree 
clf = tree.DecisionTreeClassifier(random_state=0)
clf = clf.fit(dataset.iloc[:,[2,3]], dataset.iloc[:,-1])
tree.plot_tree(clf) 
