i this we have done EDA using Bivariate and Multivariate Analysis

to import all datasets 
```bash
tips = sns.load_dataset('tips')
iris = sns.load_dataset('iris')
flights = sns.load_dataset('flights')
titanic = pd.read_csv("Titanic-Dataset.csv")
```
Scatter Plot
```bash
sns.scatterplot(x=tips['total_bill'],y=tips['tip'], hue=tips['sex'], style=tips['smoker'], size=tips['size'])
```
<img width="562" height="433" alt="5dc5c2f7-3837-4142-8bd6-c2c0b5af7643" src="https://github.com/user-attachments/assets/3ec55f3f-74ee-470a-b5e5-f77c0962e45d" />

Bar Plot
```bash
sns.barplot(x=titanic['Pclass'], y=titanic['Age'], hue=titanic['Sex'])
```
<img width="562" height="432" alt="bfb03e8f-83af-4554-9f4b-87b68e0c0f49" src="https://github.com/user-attachments/assets/ba33a189-c28c-4c18-af4c-69b2c5b1a667" />

Box Plot
```bash
sns.boxplot(x=titanic['Sex'], y=titanic['Age'], hue=titanic['Survived'])
```
<img width="562" height="432" alt="67d72d46-080c-4db6-b404-dda494b4546f" src="https://github.com/user-attachments/assets/6c8037cd-927a-4b6a-99d6-f9d214aca686" />

Dist Plot
```bash
sns.distplot(titanic[titanic['Survived']==0]['Age'],hist=False)
sns.distplot(titanic[titanic['Survived']==1]['Age'], hist=False)
```
<img width="584" height="432" alt="7f4c0f5e-d7e5-43e0-99cc-b389615cba36" src="https://github.com/user-attachments/assets/a715bf50-5801-4797-b785-2337d9443c11" />

HeatMap
```bash
s.heatmap(pd.crosstab(titanic['Pclass'], titanic['Survived']))
```
<img width="539" height="432" alt="70889f86-ce72-4f5b-b30c-533f9cc8ba92" src="https://github.com/user-attachments/assets/f73d511e-1938-47ac-a6d6-2c716566662c" />

ClusterMap
```bash
sns.clustermap(pd.crosstab(titanic['Parch'], titanic['Survived']))
```
<img width="989" height="990" alt="1b57d029-c586-4fe6-bfbf-d5bf6ee7fc84" src="https://github.com/user-attachments/assets/e759d371-0604-455f-9b73-4cb606787566" />

Pair Plot
```bash
sns.pairplot(iris, hue='species')
```
<img width="1112" height="986" alt="2cb877da-6304-4a52-acd8-3bcff0091261" src="https://github.com/user-attachments/assets/c11fdd8b-3859-4305-8b87-45550855bd94" />

Line Plot
```bash
new = flights.groupby('year').sum(numeric_only=True).reset_index()
sns.lineplot(x=new['year'],y=new['passengers'])
```
<img width="580" height="432" alt="fc6460f9-7f89-4529-9995-3a9e7c4c8318" src="https://github.com/user-attachments/assets/24df872b-6292-44d4-87be-1f1ecf8abef3" />

HeatMap 
```bash
sns.heatmap(flights.pivot_table(values='passengers', index='month', columns='year'))
```
<img width="539" height="454" alt="b9ef4808-941f-44f9-8aa8-18639c07be3e" src="https://github.com/user-attachments/assets/0d4f072b-b0d9-45ce-828f-82cec3e44974" />

ClusterMap
```bash
sns.clustermap(flights.pivot_table(values='passengers', index='month', columns='year'))
```
<img width="989" height="990" alt="1b6588c8-0450-488c-9af8-909250a5b13f" src="https://github.com/user-attachments/assets/a5ce72dc-8cb3-4eaf-94d4-df022024c997" />

