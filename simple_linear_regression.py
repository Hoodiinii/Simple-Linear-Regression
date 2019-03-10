"""
Created on Sun Mar 10 05:00:53 2019

@author: Naman
"""

#importing the libraries
import pandas as pd
import matplotlib.pyplot as plt


#importing the dataset
#change the name_of_file.csv with you file name
dataset= pd.read_csv('#name_of_file.csv')
#X is for independent variables
X= dataset.iloc[:, :-1].values
#y is for dependent variabe
y= dataset.iloc[:, 1].values


#Splitting the datatest into test set and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 1/3, random_state= 0)


#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


#fitting the simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, y_train)


#predicting the test set results
y_pred = regressor.predict(X_test)


#visualising the test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_test), color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
