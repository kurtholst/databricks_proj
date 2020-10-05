# Databricks notebook source
# MAGIC %md
# MAGIC # Data Science introduktion
# MAGIC 
# MAGIC # Lineær Regression.
# MAGIC 
# MAGIC # Use Case med ½maraton og hel maraton tider.

# COMMAND ----------

# Lineær regression helt fra bunden.

import numpy as np 
import matplotlib.pyplot as plt 

# observations 
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12]) 


plt.scatter(x, y, color = "m", 
		marker = "o", s = 30) 

# plotting the regression line 
plt.plot(x, y, color = "g") 

# putting labels 
plt.xlabel('x') 
plt.ylabel('y') 

# function to show plot 
plt.show() 
    

# Reference: https://www.geeksforgeeks.org/linear-regression-python-implementation/



# COMMAND ----------

# Beregning af koefficienter ved brug af lineær regression

def estimate_coef(x, y): 
	# number of observations/points 
	n = np.size(x) 

	# mean of x and y vector 
	m_x, m_y = np.mean(x), np.mean(y) 

	# calculating cross-deviation and deviation about x 
	SS_xy = np.sum(y*x) - n*m_y*m_x 
	SS_xx = np.sum(x*x) - n*m_x*m_x 

	# calculating regression coefficients 
	b_1 = SS_xy / SS_xx 
	b_0 = m_y - b_1*m_x 

	return(b_0, b_1) 

  
  def plot_regression_line(x, y, b): 
	# plotting the actual points as scatter plot 
	plt.scatter(x, y, color = "m", 
			marker = "o", s = 30) 

	# predicted response vector 
	y_pred = b[0] + b[1]*x 

	# plotting the regression line 
	plt.plot(x, y_pred, color = "g") 

	# putting labels 
	plt.xlabel('x') 
	plt.ylabel('y') 

	# function to show plot 
	plt.show() 

    
# plotting regression line 
plot_regression_line(x, y, b) 

# COMMAND ----------

print("Estimated coefficients:\nb_0 = {} \ nb_1 = {}".format(b[0], b[1])) 


# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#dataset = pd.read_csv('https://s3.us-west-2.amazonaws.com/public.gamelab.fun/dataset/position_salaries.csv')
dataset = pd.read_csv('https://raw.githubusercontent.com/kurtholst/databricks_proj/master/RunningTimes.csv', sep = ";") #Maraton tider
dataset.drop(['ID'], axis = 1)


X = dataset.iloc[:, 1:2].values  # features (½maraton)
y = dataset.iloc[:, 2].values    # target (full maraton)
#dataset.head(5)

print("First 5 rows")
X[0:5]

# Reference: https://towardsdatascience.com/machine-learning-polynomial-regression-with-python-5328e4e8a386


# COMMAND ----------

# Illustration af data
plt.scatter(X, y, color='red')
plt.title('Løbetider')
plt.xlabel('Halv maratontid i minutter')
plt.ylabel('Hel maratontid i minutter')
plt.show()


# COMMAND ----------

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # 80% træne og 20% teste model

# COMMAND ----------

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

print("Y = a*x + b")

print("Koefficient (a)") # Hældningskoefficienten. Fuld maraton = 2.25*(halv maraton tid i min) -6 min
print(lin_reg.coef_)

print("Intercept (b)")
print(lin_reg.intercept_)

# COMMAND ----------

# Visualizing the Linear Regression results
def viz_linear():
    plt.scatter(X, y, color='red')
    plt.plot(X, lin_reg.predict(X), color='blue')
    plt.title('Løbetider')
    plt.xlabel('Halv maratontid i minutter')
    plt.ylabel('Hel maratontid i minutter')

    plt.show()
    return
viz_linear()

# COMMAND ----------

# Evaluering af model
# R^2 (coefficient of determination) regression score function.
# Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). 
# A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.
from sklearn.metrics import r2_score
y_pred = lin_reg.predict(X_test)

print("r2 score")
r2_score(y_test, y_pred)

# COMMAND ----------

# Evaluering med RMSE
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)

# COMMAND ----------

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4) # Antal polynomier
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

print("Koefficienter til polynomium:")
print(pol_reg.coef_)

print("Intercept :")
print(pol_reg.intercept_)

print("Polynomium:")
print(pol_reg.rank_)


# COMMAND ----------

# Visualizing the Polymonial Regression results
def viz_polymonial():
    plt.scatter(X, y, color='red')
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title('Løbetider')
    plt.xlabel('Halv maratontid i minutter')
    plt.ylabel('Hel maratontid i minutter')
    plt.show()
    return
viz_polymonial()

# COMMAND ----------

# Evaluering af poly-model
#y_pred = pol_reg.predict((X))
poly_features = PolynomialFeatures(degree=4)
  
# predicting on test data-set
y_pred  = pol_reg.predict(poly_features.fit_transform(X_test))
  
# evaluating the model on training dataset
rmse_train = np.sqrt(mean_squared_error(y_test, y_pred))
r2_train = r2_score(y_test, y_pred)

print("r2 score")
print(r2_train)

print("rmse")
print(rmse_train)

# COMMAND ----------

# Predicting on new data
half_marathon = 100
# Predicting a new result with Linear Regression
print("Linear Regression: ")
print(lin_reg.predict([[half_marathon]]))

print("Fuld maraton = 2.25*(halv maraton tid i min) -6 min")
print("dvs. 2,24912973 * 100 - 6.005069013188233 = ")
print(2.24912973 * 100 - 6.005069013188233)

print("")

# Predicting a new result with Polymonial Regression
print("Polynomial Regression:")
pol_reg.predict(poly_reg.fit_transform([[half_marathon]]))


# COMMAND ----------

# MAGIC %md 
# MAGIC # Simpelt eksempel

# COMMAND ----------

import matplotlib.pyplot as plt

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

plt.scatter(x, y)
plt.show()

# COMMAND ----------

import numpy
import matplotlib.pyplot as plt

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

myline = numpy.linspace(1, 22, 100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()

# COMMAND ----------

import numpy
from sklearn.metrics import r2_score

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

print(r2_score(y, mymodel(x)))

# COMMAND ----------

import numpy
from sklearn.metrics import r2_score

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

#Predict
speed = mymodel(17)
print(speed)

# COMMAND ----------

import numpy
import matplotlib.pyplot as plt

x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

myline = numpy.linspace(2, 95, 100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()

# COMMAND ----------

