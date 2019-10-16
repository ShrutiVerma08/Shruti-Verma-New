import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('C:/Users/Shruti Verma/Downloads/tcdml1920-income-ind/tcd ml 2019-20 income prediction training (with labels).csv')
len(df)
df = df.rename(index=str, columns={"Income in EUR": "Income"})

Test_Data=pd.read_csv('C:/Users/Shruti Verma/Downloads/tcdml1920-income-ind/tcd ml 2019-20 income prediction test (without labels).csv')
len(Test_Data)

#Y = data.iloc[:,11].values
df=pd.concat([df,Test_Data], sort=False)
df = df.drop("Instance", axis=1)     #Drop column that has no relevance



df['University Degree'] = df['University Degree'].replace('0', "No")
df['Hair Color'] = df['Hair Color'].replace('0', "Unknown")
age_mean = df['Age'].mean()
Record_mean=df['Year of Record'].mean()
df['Age'] = df['Age'].replace('#N/A',age_mean )
df['Year of Record'] = df['Year of Record'].replace('#N/A',Record_mean )
   
df['Gender'] = df['Gender'].replace('0', "Other")
df['Gender'] = df['Gender'].replace('#N/A', "Other")
#data = data[data.Age < 85]
#X = train.drop("Income in EUR", axis=1)
 
#df = df.dropna()
#df.isnull().values.any()
#df.isnull().sum()
#X = data.iloc[:,:-1].values
df = df.rename(index=str, columns={"Body Height [cm]": "Height"})
#data = data[data.Height > 120]
#data = data[data.Height < 235]
#X = data.iloc[:,:-1].values
data2 = pd.get_dummies(df, columns=["Gender"])
data2 = pd.get_dummies(data2, columns=["Country"])
data2 = pd.get_dummies(data2, columns=["Profession"])
data2 = pd.get_dummies(data2, columns=["University Degree"])
data2 = pd.get_dummies(data2, columns=["Hair Color"])

train = data2[0:111994]
train = train.dropna()

Y = train.Income  
X = train.drop("Income", axis=1)





#X = data1.iloc[:,:-1].values

from sklearn.linear_model import LinearRegression
LR = LinearRegression()


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)
model = LR.fit(X_train, Y_train)
len(X)
len(Y)
len(X_test)

#
 
Y_pred = LR.predict(X_test)



from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y_test, Y_pred)
mse

import math
rmse = math.sqrt(mse)
rmse


test = data2[111993:]
X_test = test.drop("Income", axis=1)
#Y = test.Income
age_mean = X_test['Age'].mean()
Record_mean=X_test['Year of Record'].mean()
X_test['Age'] = X_test['Age'].replace(pd.np.nan,age_mean )
X_test['Year of Record'] = X_test['Year of Record'].replace(pd.np.nan,Record_mean )

y1 = LR.predict(X_test)
y1 = pd.DataFrame(y1)
y1.to_csv("C:/Users/Shruti Verma/Downloads/tcdml1920-income-ind/submit.csv")
