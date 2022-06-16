####################################################################################
#                                                                                  #
#   This program takes the csv file of cleaned data, called 'coupon.csv' and       #  
#   performs various ML algorithms to test which ones can possibly reach the       #
#   desired accuracy of >= 70%.  This will print out the following results from    #
#   the following ML classifications                                               #
#       Perceptron                                                                 #
#       Logistic Regression                                                        #
#       Support Vector Machine (pick one version)                                  #
#       Decision Tree Learning                                                     #
#       Random Forest                                                              #
#       K-Nearest Neighbor                                                         #
#                                                                                  #
####################################################################################



#Dataframe generation/manipulation + computational packages
import pandas as pd
import numpy as np

#data visualization packages
import matplotlib.pyplot as plt

#machine leanring packages
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

coupon = pd.read_csv("coupon.csv")  #import csv document created from part 1
coupon = coupon.drop(['Unnamed: 0'], axis=1) #unknown column created from import.  This drops it

X = coupon.iloc[:,:24]  #slices dataframe to include the first 24 columns and all its features
y = coupon.iloc[:,-1] #slices the last column.  This is the "Class" column

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0) #parses data into test sets and training sets

sc = StandardScaler()  #standardizing package.  Assigning it variable
sc.fit(X_train) #compute the required transformation.  ONLY do this on the FEATURES TRAINING data and NOT Class

X_train_std = sc.transform(X_train) #standardizes the features training data
X_test_std = sc.transform(X_test) #standardizes the features test data

X_combined_std = np.vstack((X_train_std, X_test_std)) #combines the standardized features data into one large array by vertical stacking
X_combined = np.vstack((X_train, X_test)) #perfoms same task as above, but in the non-standardized feature data
y_combined = np.hstack((y_train, y_test)) #performs a horizontal stack of the one dimnesional array for the Class


# create the classifier and train it
ppn = Perceptron(max_iter=20, tol=1e-5, eta0=0.001, fit_intercept=True, random_state=0, verbose=False)#instantiates the Perceptron algorithm
ppn.fit(X_train_std, y_train) # do the training
y_pred = ppn.predict(X_test_std) # now try with the test data
y_combined_pred = ppn.predict(X_combined_std) # prediction of the combined feature set
# show the results
print('\n')
print('Perceptron Test Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Perceptron Combined Accuracy: %.2f\n' % accuracy_score(y_combined, y_combined_pred))

# create the classifier and train it
lr = LogisticRegression(C=1, solver='liblinear', multi_class='ovr', random_state=0) #instantiates the Logistic Regression algorithm
lr.fit(X_train_std, y_train) # do the trainin
y_pred = lr.predict(X_test_std) # work on the test data
y_combined_pred = lr.predict(X_combined_std) # prediction of the combined feature set
# show the results
print('Logistic Regression Test Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Logistic Regression Combined Accuracy: %.2f\n' % accuracy_score(y_combined, y_combined_pred))

# create the classifier and train it
#svm = SVC(kernel='linear', C=1, random_state=0) 
#svm.fit(X_train_std, y_train) # do the training
#y_pred = svm.predict(X_test_std) # work on the test data
#y_combined_pred = svm.predict(X_combined_std)
#print('SVM Test Accuracy: %.2f' % accuracy_score(y_test, y_pred))
#print('SVM Combined Accuracy: %.2f\n' % accuracy_score(y_combined, y_combined_pred))

# create the classifier and train it
svm = SVC(kernel='rbf', tol=1e-3, random_state=0, gamma=.1, C=1.0, verbose=False) #instantiates the SVM algorithm
svm.fit(X_train_std, y_train) # do the trainin'
y_pred = svm.predict(X_test_std) # work on the test data
y_combined_pred = svm.predict(X_combined_std)# prediction of the combined feature set
# show the results
print('SVM 2 Test Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('SVM 2 Combined Accuracy: %.2f\n' % accuracy_score(y_combined, y_combined_pred))

# create the classifier and train it
tree = DecisionTreeClassifier(criterion='entropy',max_depth=5 ,random_state=0) #instantiates theDecision Tree algorithm
tree.fit(X_train,y_train) # do the trainin'
y_pred = tree.predict(X_test) # work on the test data
y_combined_pred = tree.predict(X_combined)# prediction of the combined feature set
# show the results
print('Decision Tree Test Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Decision Tree Combined Accuracy: %.2f\n' % accuracy_score(y_combined, y_combined_pred))

# create the classifier and train it
# n_estimators is the number of trees in the forest
# the entropy choice grades based on information
forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=0, n_jobs=4) #instantiates the Random Forest algorithm
forest.fit(X_train,y_train) # do the trainin'
y_pred = forest.predict(X_test) # see how we do on the test data
y_combined_pred = forest.predict(X_combined) # see how we do on the combined data
# show the results
print('Random Forest Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Random Forest Combined Accuracy: %.2f\n' % accuracy_score(y_combined, y_combined_pred))

# create the classifier and fit it
# where p=2 specifies sqrt(sum of squares). (p=1 is Manhattan distance)
knn = KNeighborsClassifier(n_neighbors=30,p=1,metric='minkowski')#instantiates the K Nearest Neighbor algorithm
knn.fit(X_train_std,y_train) # do the trainin'
y_pred = knn.predict(X_test_std)# work on the test data
y_combined_pred = knn.predict(X_combined_std) # prediction of the combined feature set
# show the results
print('KNN Test Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('KNN Combined Accuracy: %.2f\n' % accuracy_score(y_combined, y_combined_pred))

