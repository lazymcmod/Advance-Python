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
26. Power Transformer
 

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

Power Transformer

Necessary Library
```bash
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PowerTransformer
```
Import Dataset
```bash
df = pd.read_csv("https://raw.githubusercontent.com/campusx-official/100-days-of-machine-learning/refs/heads/main/day31-power-transformer/concrete_data.csv")
```

Apply Regression Without Any Transformer
```bash
lr = LinearRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

r2_score(y_test,y_pred)
```
plotting the distplot without any transformation
```bash
for col in X_train.columns:
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    sns.histplot(X_train[col])
    plt.title(col)

    plt.subplot(122)
    stats.probplot(X_train[col], dist="norm",plot=plt)
    plt.title(col)

    plt.show()
```
<img width="1160" height="393" alt="ac390788-b0f4-411e-b7c5-87c803c77a01" src="https://github.com/user-attachments/assets/42a1d6da-7487-4783-a558-4a79a967283f" />
<img width="1160" height="393" alt="3fc4aed8-6e68-45f6-8cf1-e477aa39bba3" src="https://github.com/user-attachments/assets/07f695fb-d2eb-4e6e-a304-22e596dc2d80" />
<img width="1160" height="393" alt="332e6024-ca7d-4817-9590-62992163329b" src="https://github.com/user-attachments/assets/fad3712c-35e6-4187-8dfd-a13ca9508a71" />
<img width="1160" height="393" alt="be195a04-91be-4103-93ab-1d76e24728ee" src="https://github.com/user-attachments/assets/b6d6d65a-672e-43b5-afe8-24906dcc0e39" />
<img width="1160" height="393" alt="c3aabbf0-a6f3-490d-9855-6a2a5d5ff378" src="https://github.com/user-attachments/assets/5239910b-ecb8-452c-870e-4d13e41a313b" />
<img width="1160" height="393" alt="9e7ab90d-69b6-49f1-bd08-078e5dfc181f" src="https://github.com/user-attachments/assets/d0030431-f235-4a73-99fa-b61a77578a53" />
<img width="1160" height="393" alt="7d2cb43d-0868-4926-ac3b-e815feaf7a53" src="https://github.com/user-attachments/assets/9b05daaf-0478-4028-8a99-1eec3938a0d3" />
<img width="1160" height="393" alt="db61c1c8-d02e-4945-8921-43f5c9c2e35b" src="https://github.com/user-attachments/assets/bc9e10fa-f0f0-4ace-bb4a-6119f9d0869d" />

 Apply Box-Cox Transform
 ```bash
pt  =PowerTransformer(method='box-cox')

X_train_Transformed = pt.fit_transform(X_train+0.000001)
X_test_Transformed = pt.fit_transform(X_test+0.000001)

pd.DataFrame({'cols':X_train.columns,'box_cox_lambda':pt.lambdas_})
```
Apply Linear Regression on Transformed Data
```bash
lr = LinearRegression()
lr.fit(X_train_Transformed,y_train)

y_pred2 = lr.predict(X_test_Transformed)
r2_score(y_test,y_pred2)
```
Before and after comparsion for Box-Cox plot
```bash
X_train_Transformed = pd.DataFrame(X_train_Transformed,columns=X_train.columns)
for col in X_train_Transformed.columns:
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    sns.histplot(X_train[col])
    plt.title(col)

    plt.subplot(122)
    sns.histplot(X_train_Transformed[col])
    plt.title(col)

    plt.show()
```
<img width="1160" height="393" alt="9d645afd-6f7d-4896-bb96-154257a35aa4" src="https://github.com/user-attachments/assets/d811859b-99a5-4347-a993-22dd5937dab4" />
<img width="1160" height="393" alt="a08af06e-9b3f-4150-aacb-89f21394d08e" src="https://github.com/user-attachments/assets/c9eccdbe-827e-4eed-adc7-11ba7e4f8749" />
<img width="1160" height="393" alt="483a21f9-e5a7-4c98-b208-9bd241fe595b" src="https://github.com/user-attachments/assets/041c5032-fe79-47c6-ad93-035a72812526" />
<img width="1160" height="393" alt="cd7b211b-f4ad-4d44-a5ab-28be454946f8" src="https://github.com/user-attachments/assets/15c58591-7772-459a-adde-a4bdc763cf18" />
<img width="1160" height="393" alt="07f644a5-a172-430d-9a84-9e018fa73a1f" src="https://github.com/user-attachments/assets/0bff7336-1d2b-472c-92a9-d79db910e2d3" />
<img width="1160" height="393" alt="4e3e7b6b-356e-43b2-8e92-a035efd23e80" src="https://github.com/user-attachments/assets/1488121d-6202-4eb8-ab32-62fb823f3cf2" />
<img width="1160" height="393" alt="67db8f4f-325b-466d-ab7a-dc5668cd6be2" src="https://github.com/user-attachments/assets/429e8c1e-139c-4d78-83e4-4710f9290701" />
<img width="1160" height="393" alt="3933a12f-6023-490f-adf5-eb6b8c648ce9" src="https://github.com/user-attachments/assets/94197ae7-109d-4ab9-b5ed-b6f7f0065e28" />

Apply Yeo-Johnson Transform
```bash
pt1 = PowerTransformer()

X_train_Transformed2 = pt1.fit_transform(X_train)
X_test_Transformed2 = pt1.fit_transform(X_test)

lr = LinearRegression()
lr.fit(X_train_Transformed2,y_train)

y_pred3 = lr.predict(X_test_Transformed2)

print(r2_score(y_test,y_pred3))

pd.DataFrame({'cols':X_train.columns,'Yeo_Johnson_lambdas':pt1.lambdas_})
```
Before and after comparision for Yeo-Johnson
```bash
for col in X_train_Transformed2.columns:
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    sns.histplot(X_train[col])
    plt.title(col)

    plt.subplot(122)
    sns.histplot(X_train_Transformed2[col])

    plt.show()
```
<img width="1160" height="393" alt="3f780038-7781-4d47-979e-2a74d8203399" src="https://github.com/user-attachments/assets/021ab5e9-9e7f-4c9d-b5db-4357080b3275" />
<img width="1167" height="393" alt="5970bdc6-37c5-41b7-96ab-df91d8e7b0e2" src="https://github.com/user-attachments/assets/f338f504-05db-43fa-adf4-3a5a5d2e80fa" />
<img width="1160" height="393" alt="dbd5cba5-4804-4b12-a5a7-40044a0f9d2e" src="https://github.com/user-attachments/assets/fed41960-ee04-4b09-a668-298e3412d80e" />
<img width="1160" height="393" alt="4226bc1f-2ade-4b3a-b436-0ef8a8e5bc40" src="https://github.com/user-attachments/assets/bc411d0e-e3ec-40f4-927c-2c3ed7ef5345" />
<img width="1160" height="393" alt="bf577071-f5c5-42f2-b86c-9d9fc4486317" src="https://github.com/user-attachments/assets/a72772e2-2f9c-44d1-9739-e04afe6230b2" />
<img width="1160" height="393" alt="79db7927-0220-4f54-be26-19d9f787d759" src="https://github.com/user-attachments/assets/cdaea7a3-b58f-4402-a34a-1980dbac1715" />
<img width="1160" height="393" alt="b4d7bb29-feb7-439a-968f-fbde41f95bd0" src="https://github.com/user-attachments/assets/092c18ce-b7d8-49c4-af92-7eb4be4384ac" />
<img width="1160" height="393" alt="f39328ab-4e0b-424f-8573-33d693efceee" src="https://github.com/user-attachments/assets/5bb81206-0f83-4ec7-9aa1-86f9eae7f719" />


   









