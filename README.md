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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
dataset=pd.read_csv('Placement.csv')
dataset
dataset=dataset.drop('sl_no',axis=1)
dataset["gender"]=dataset["gender"].astype("category")
dataset["ssc_b"]=dataset["ssc_b"].astype("category")
dataset["hsc_b"]=dataset["hsc_b"].astype("category")
dataset["degree_t"]=dataset["degree_t"].astype("category")
dataset["workex"]=dataset["workex"].astype("category")
dataset["specialisation"]=dataset["specialisation"].astype("category")
dataset["status"]=dataset["status"].astype("category")
dataset["hsc_s"]=dataset["hsc_s"].astype("category")
dataset.dtypes
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset.info()
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
print(X.shape)
print(Y.shape)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
clf = LogisticRegression()
clf.fit(X_train,Y_train)
clf.score(X_test,Y_test)
Y_pred=clf.predict(X_test)
print(Y_pred)
from sklearn.metrics import confusion_matrix,accuracy_score
cf=confusion_matrix(Y_test,Y_pred)
print(cf)
accuracy=accuracy_score(Y_test,Y_pred)
print(accuracy)

```

## Output:
![Screenshot 2024-10-09 103356](https://github.com/user-attachments/assets/ffac10b3-c92c-4c7f-85b2-1f77a2010ab8)

![Screenshot 2024-10-09 103427](https://github.com/user-attachments/assets/2624884e-3053-42d1-8072-75516f505650)

![Screenshot 2024-10-09 103433](https://github.com/user-attachments/assets/317ab997-3add-431f-8c3b-e136cc114cf8)

![Screenshot 2024-10-09 103440](https://github.com/user-attachments/assets/f56e442d-5c4b-485e-a20b-8b617a8361fc)

![Screenshot 2024-10-09 103445](https://github.com/user-attachments/assets/5bcb4842-08a6-4125-97b9-badea00f9490)

![Screenshot 2024-10-09 103451](https://github.com/user-attachments/assets/c8dad4d3-7df1-4035-aedf-18321ed6194b)

![Screenshot 2024-10-09 103456](https://github.com/user-attachments/assets/5705dca6-5054-44b4-9288-0bb43dded1bf)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
