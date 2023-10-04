# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

   1. Import pandas, numpy and mathplotlib.pyplot
    2.Trace the best fit line and calculate the cost function
    3.Calculate the gradient descent and plot the graph for it
    4.Predict the profit for two population sizes.


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: THRIKESWAR P
RegisterNumber: 212222230162

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  """
  Take in a numpy array X, y , theta and generate the cost fuction in a linear regression model
  """
  m=len(y)
  h=X.dot(theta)  # length of training data
  square_err=(h-y)**2 

  return 1/(2*m) * np.sum(square_err)  
  
  data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)   # function call

def gradientDescent(X,y,theta,alpha,num_iters):
  
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions - y))
    descent=alpha * 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))

  return theta, J_history
  
theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iternations")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Polpulation of City (10,000s)")
plt.ylabel("Profit (10,000s)")
plt.title("Profit Prediction")

def predict(x,theta):
  predictions= np.dot(theta.transpose(),x)
  return predictions[0]
  
 predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0))) 
*/ 
```

## Output:

![1](https://github.com/Naveensrinivasan07/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475891/60380a51-46c1-4872-ab50-b3ba57dba68d)
![2](https://github.com/Naveensrinivasan07/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475891/66ad3657-3281-4031-8d68-e6d7c5106f3a)
![3](https://github.com/Naveensrinivasan07/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475891/b6745172-9c5f-4551-96f8-f619f14b689d)
![4](https://github.com/Naveensrinivasan07/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475891/427c046a-0751-494b-a52d-3b54e2cf78c8)
![5](https://github.com/Naveensrinivasan07/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475891/aa3fb5ff-2178-4625-8cfd-52547c680e71)
![6](https://github.com/Naveensrinivasan07/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475891/77a26c0b-a8c6-4e56-9e68-7c2f86a69dc4)
![7](https://github.com/Naveensrinivasan07/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475891/6c2c30db-f73e-494d-a747-cf591d3f6146)
![8](https://github.com/Naveensrinivasan07/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475891/e2b8b243-3b35-4c3b-b6fe-3622bbefda37)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming
