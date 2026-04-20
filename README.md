You can use this to learn python from very basic level

In this Jupyter Notebook we have all concept of Python 
1. Function
2. Oops (Object Oriented Programming)
3. Encapusulation
4. Inheritance
5. Polymorphism
6. Abstraction
7. Error Handling
8. Expection
9. Bank Management Mini Project
10. Introduction to Machine Learning
11. Supervised Machine Learning
12. Confusion Metrics
13. Linear Regression Pratical Implementation
14. EDA
15. Ridge and Lasso Implementation
16. Lasso Regression
17. Decision Tree Implementation
18. Gradient Desecent
19. EDA using Bivariate and Multivariate Analysis
20. Pandas Profiling
21. Encoding Categorical Data
22. One Hot Encoding
23. Column Transformer in Machine Learning
24. Machine Learning Pipelines A-Z\
25. Transformer
 

To Install Require Library
Numpy
```bash
pip install Numpy
```
Pandas
```bash
pip install Pandas
```
Matplotlib
```bash
pip install matplotlib
```
Sklearn
```bash
pip install scikit-learn
```
Seaborn
```bash
pip install seaborn
```
<img width="1058" height="986" alt="download" src="https://github.com/user-attachments/assets/ebd7b145-a0c1-412d-80ec-6bc741c84f68" />

Sklearn (Train-test-split)
```bash
from sklearn.model_selection import train_test_split
```
Sklearn (Linear Regression)
```bash
from sklearn.linear_model import LinearRegression
```
Sklearn (Logistic Regression)
```bash
from sklearn.linear_model import LogisticRegression
```
Sklearn (KNN)
```bash
from sklearn.neighbors import KNeighborsClassifier
```
Sklearn (Decision Tree Classifier)
```bash
from sklearn.tree import DecisionTreeClassifier
```
Sklearn Metrics (accuracy_score, precision_score, recall_score, f1_score)
```bash
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
```
Sklearn Metrics (Confusion Metrics)
```bash
from sklearn.metrics import confusion_matrix
```
Sklearn (Tree)
```bash
from sklearn import tree
```
<img width="794" height="790" alt="download (1)" src="https://github.com/user-attachments/assets/c917db47-854d-4c4f-8711-c863ce03d126" />


Seaborn 
```bash
import seaborn as sns
```
Matplotlib
```bash
from matplotlib import pyplot as plt
```
Pandas 
```bash
import pandas as pd
```
Numpy
```bash
import numpy as np
```
All library need to add before doing Transformer

```bash
import pandas as pd
import numpy as np

import scipy.stats as stats

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
```
To add dataset

```bash
df = pd.read_csv("Titanic-Dataset.csv",usecols=['Age', 'Fare', 'Survived'])
```
Apply TrainTestSplit

```bash
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
```

Graphs of Age

```bash
plt.figure(figsize=(14,4))
plt.subplot(121)
sns.distplot(X_train['Age'])
plt.title("Age PDF")

plt.subplot(122)
stats.probplot(X_train['Age'],dist="norm", plot=plt)
plt.title("Age QQ Plot")

plt.show()
```
<img width="1165" height="393" alt="7e216e55-81c9-44f3-a2db-39b474d0a3aa" 
src="https://github.com/user-attachments/assets/4d2efe2b-1fc7-42d9-a4b8-abd90ce54723" />

Graph of Fare

```bash
plt.figure(figsize=(14,4))
plt.subplot(121)
sns.distplot(X_train['Age'])
plt.title("Age PDF")

plt.subplot(122)
stats.probplot(X_train['Age'],dist="norm", plot=plt)
plt.title("Age QQ Plot")

plt.show()
```
<img width="1165" height="393" alt="7e216e55-81c9-44f3-a2db-39b474d0a3aa" src="https://github.com/user-attachments/assets/83998748-69c3-41ba-8b33-3dc2249035ab" />


Apply LogisticRegression and DecisionTreeClassifier

```bash
clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(X_train_Transformed,y_train)
clf2.fit(X_train_Transformed,y_train)

y_pred = clf.predict(X_test)
y_pred1 = clf2.predict(X_test)

print("Accuracy LR",accuracy_score(y_test,y_pred))
print("Accuracy DT",accuracy_score(y_test,y_pred1))
```

Age Before Log

```bash
plt.figure(figsize=(14,4))

plt.subplot(121)
stats.probplot(X_train['Fare'],dist="norm",plot=plt)
plt.title("Age before Log")

plt.subplot(122)
stats.probplot(X_train_Transformed['Fare'],dist="norm",plot=plt)
```
<img width="1172" height="393" alt="b2808917-4ea0-4627-a7ef-44967ab710e0" src="https://github.com/user-attachments/assets/1968d847-7179-48f7-9372-93165c903366" />

Apply Transformer

```bash
def apply_transformer(transformer):
    X = df.iloc[:,1:3]
    y = df.iloc[:,0]

    trf = ColumnTransformer([('log',FunctionTransformer(transformer),['Fare'])],remainder='passthrough')

    X_trans = trf.fit_transform(X)

    clf = LogisticRegression()

    print('Accuracy',np.mean(cross_val_score(clf,X_trans,y,scoring='accuracy',cv=10)))

    plt.figure(figsize=(14,4))

    plt.subplot(121)
    stats.probplot(X['Fare'],dist="norm",plot=plt)
    plt.title("Fare Before Transform")

    plt.subplot(122)
    stats.probplot(X_trans[:,0],dist="norm",plot=plt)
    plt.title("Fare After Transform")

    plt.show()
```
1. ```bash
   apply_transformer(lambda x: X**2)
   ```
   Accuracy 0.6577403245942571
   <img width="1172" height="393" alt="20ffa447-721a-4551-9768-bb58ebe5b614" src="https://github.com/user-attachments/assets/76f05fc4-428c-4684-a212-019129a7b251" />

2. ```bash
   apply_transformer(lambda x: X**3)
   ```
   Accuracy 0.6251685393258427
   <img width="1172" height="393" alt="9437e3c2-ab18-46df-b99c-39e1ec89b0ad" src="https://github.com/user-attachments/assets/9feb1855-1550-4bd1-843c-592ca02d4057" />

3. ```bash
   apply_transformer(lambda x: X**1/2)
   ```
   Accuracy 0.6589013732833957
   <img width="1172" height="393" alt="3ee342dc-f47c-47a7-ab0e-779ab27a6787" src="https://github.com/user-attachments/assets/52d42fb7-d2a0-4e8a-89a2-a8094fa84177" />

4. ```bash
   apply_transformer(lambda x: 1/(x+0.1))
   ```
   Accuracy 0.61729088639201
   <img width="1172" height="393" alt="5ee327cf-fbd8-4b7d-b614-cbe8273a5108" src="https://github.com/user-attachments/assets/f6e0567d-446f-45a7-9f11-fc256fffdfd7" />

   
   

   


   






