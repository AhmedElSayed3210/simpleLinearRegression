# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2, random_state = 0)


from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train, y_train)

y_pre=regression.predict(x_test)

# train set
plt.scatter(x_train , y_train, color='red')
plt.plot(x_train, regression.predict(x_train) , color='blue')
plt.title(' Salary VS Experince(traning set)')
plt.xlabel('years pf erperince')
plt.ylabel('salary')
plt.show()

# test set
plt.scatter(x_test , y_test, color='red')
plt.plot(x_train, regression.predict(x_train) , color='blue')
plt.title(' Salary VS Experince(test set)')
plt.xlabel('years pf erperince')
plt.ylabel('salary')
plt.show()
