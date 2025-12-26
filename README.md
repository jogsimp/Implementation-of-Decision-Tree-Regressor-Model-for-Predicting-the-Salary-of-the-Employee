# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Joshua Abraham Philip A
RegisterNumber: 25013744

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("salary.csv")

X = df[['Level']]
y = df['Salary']

regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X, y)

y_pred = regressor.predict(X)
print("Predicted salaries:", y_pred)

level = np.array([[6.5]])
predicted_salary = regressor.predict(level)
print(f"Predicted Salary for level {level[0][0]}: {predicted_salary[0]}")

X_grid = np.arange(min(X.values), max(X.values) + 0.01, 0.01)
X_grid = X_grid.reshape(-1, 1)

plt.scatter(X, y, color='red', label='Actual Salary')
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Decision Tree Prediction')
plt.title('Decision Tree Regression: Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.show()


*/
```

## Output:
<img width="566" height="398" alt="image" src="https://github.com/user-attachments/assets/a32c0e39-c814-4fcc-8aeb-fe8ef91dcd7a" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
