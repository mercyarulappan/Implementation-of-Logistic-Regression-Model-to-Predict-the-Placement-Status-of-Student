# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results. 

## Program:
```python
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: MERCY A
RegisterNumber: 212223110027
*/
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data=fetch_california_housing()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.info())

X = df.drop(columns=['AveOccup','target'])
Y = df[['AveOccup','target']]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state = 1)

# Scale the features and target variable
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

# Converting the data in range
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.fit_transform(X_train)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.fit_transform(Y_train)


# Initialize the SGDRegressor
sgd = SGDRegressor(max_iter=1000 ,tol=1e-3)

# Use MultiOutputRegressor to handle multiple output variables
multi_output_sgd = MultiOutputRegressor(sgd)

# Train the model
multi_output_sgd.fit(X_train,Y_train)

# Predict on the test data
Y_pred = multi_output_sgd.predict(X_test)

# inverse transform the predictions to get them back to the original scale
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:",mse)

# Optionally, print some predictions 
print("\nPredictions:\n",Y_pred[:5])    ## Print first 5 predictions

```

## Output:

![Screenshot 2024-09-04 140359](https://github.com/user-attachments/assets/c9d360be-f852-4718-8b56-1dab1a6f0aec)

![Screenshot 2024-09-04 135720](https://github.com/user-attachments/assets/f13c93d0-8cda-47fd-a59e-157c9a7aded9)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
