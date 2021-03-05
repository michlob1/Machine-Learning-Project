#### Support Vector Machine (SVM)

# Importing the libraries
import numpy as np #to work with arrays 
import matplotlib.pyplot as plt #to plot charts
import pandas as pd #import db and create matrix of features


# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values #matrix of features (X) 
y = dataset.iloc[:, -1].values #dependant variable vector (y)


# Splitting the dataset into the Training set and Test set

"""
It is necessary to split the data before applying features scaling to prevent information 
leakage on the test set (which you're not supposed to have until the training).
                         
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    #matrix of features, dependent variable vector, 25% test, always the same split 
print(X_train)
print(y_train)
print(X_test)
print(y_test)


# Feature Scaling

"""
To avoid some features being dominated by other features in such a way that the dominated ones are 
not even considered by some machinery model (it is needed for just some machine learning models)

"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
    #fit_transform: fit gets the mean and sd of the data and transforms the data
X_test = sc.transform(X_test)
    #trasform: applies the same trasnformation done to X_train 
print(X_train)
print(X_test)


# Training the SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0) #Parameters of the SVM model 
classifier.fit(X_train, y_train) #train of the SVM model


# Predicting a new result
print(classifier.predict(sc.transform([[30,87000]])))

""""
Important note: The value of the features must be entered in a double pair of square 
brackets,  due to the "predict" method always expects a 2D array as the format of its inputs. 

Simply put:

12 --> scalar
[12] --> 1D array
[[12]] --> 2D array

"""


# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) 
    #creates an array to contrast the predicted values (column 0) and the real values (column 1)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

"""
     _____ _____ 
    |  0  |  1  |
 --- ----- -----  TN: True negative results 
| 0 | TN  | FP  | FP: False positive results 
 --- ----- -----  FN: False negative results 
| 1 | FN  | TP  | TP: True positive results
 --- ----- -----
 
"""

accuracy_score(y_test, y_pred)
    #accuracy = (TP+TN/ All) *100%


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train #inverse_transform: data as originally was
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25), 
                     #np.arange:return evenly spaced values within a given interval
                     #to create a very dense grid to visualize the values 
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()