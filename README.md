# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. mport the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph. 

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: Mythili D
RegisterNumber:  212222040104
```
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city(10,000s")
plt.ylabel("Profit($10,000")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  m=len(y)
  h=X.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
  
data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  j_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    j_history.append(computeCost(X,y,theta))
  return theta,j_history  
  
theta,j_history=gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(j_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000)s")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions= np.dot(theta.transpose(),x)
    return predictions[0]
    
predict1=predict(np.array([1,3.5]),theta)*10000##Co
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:
## PROFIT PREDICTION:
![image](https://github.com/Mythilidharman/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119104110/2f50631c-b491-469b-8513-f59c359a6714)

## compute cost
![image](https://github.com/Mythilidharman/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119104110/225aa50b-9f8f-4e3a-a3bc-cdf7b2495f79)

## h(x) value
![image](https://github.com/Mythilidharman/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119104110/7e257f7d-01ba-47c7-bc4a-4ced167f1484)

## Cost function using gradient descent Graph
![image](https://github.com/Mythilidharman/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119104110/a8fd1501-3967-4b3b-ae1a-fc60e49707e4)
## Profit prediction
![image](https://github.com/Mythilidharman/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119104110/843971b7-9cb1-451e-be4d-e61a77893eda)

## Profit for the Population 35,000
![image](https://github.com/Mythilidharman/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119104110/4937bd04-252d-420b-a49e-b31957949ca9)
## Profit for the Population 70,000
![image](https://github.com/Mythilidharman/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119104110/2a2d6404-30a6-4bc8-a9b4-a985f807c62d)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
