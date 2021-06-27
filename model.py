import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle 
df= pd.read_csv("/content/hiring.csv")


#fill score with null
df['test_score'].fillna(df['test_score'].mean(),inplace=True)

df['experience'].fillna(0,inplace=True)

X = df.iloc[:,:3]
Y = df.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.1,random_state=5)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
y_pred= model.predict(x_test)
y = model.predict([[5,8,7]])
y

pickle.dump(model, open("model.pkl","wb"))
