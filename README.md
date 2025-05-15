# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data. 
3. Use modelselection and Countvectorizer to preditct the values. 
4. Find the accuracy and display the result.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: KARTHIKEYAN S
RegisterNumber:  212224230116
*/
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
```
```
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
```
```
data = pd.read_csv('spam.csv',encoding='windows-1252')
df = pd.DataFrame(data)
df
```
```
df.isnull().sum()
```
```
x = df['v1'].values
x
```
```
y = df['v2'].values
y
```
```
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
```
```
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
```
```
svc = SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
y_pred
```
```
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy Score: {accuracy}")
```

## Output:
### Dataset
![Screenshot 2025-05-15 101558](https://github.com/user-attachments/assets/9cd792fa-eca3-4cd9-97da-23c8573d02b5)

### X Value
![Screenshot 2025-05-15 101635](https://github.com/user-attachments/assets/840622c5-fcbe-4f25-ae6f-b77ddcbf3385)

#### Y Value
![Screenshot 2025-05-15 101704](https://github.com/user-attachments/assets/a4291153-b321-49d8-8c55-0943892e1f80)

### Predicted Value
![Screenshot 2025-05-15 101717](https://github.com/user-attachments/assets/ad6a27d7-d6ec-4ecf-a944-5ee4bde6a679)


### Accuracy Score
![Screenshot 2025-05-15 101732](https://github.com/user-attachments/assets/90097f83-d40a-4ff1-8423-bb6d3d8acc13)






## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
