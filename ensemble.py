import pandas as pd
df=pd.DataFrame()
df = pd.read_csv(r'C:\Users\HP\OneDrive\Documents\research_theory\omnicron.csv',index_col='Date_reported',parse_dates=True)
data=pd.read_csv(r'C:\Users\HP\OneDrive\Documents\research_theory\omnicron.csv',parse_dates=True)
#df.set_index('Dat',inplace=True)
df.index=df.index.to_period('D')
print(df.tail())
df.columns = ['New_cases']
#df.plot(figsize=(12,8))
df['Sale_1day']=df['New_cases'].shift(+1)
df['Sale_2day']=df['New_cases'].shift(+2)
df['Sale_3day']=df['New_cases'].shift(+3)
df=df.dropna()

import sklearn
from sklearn import linear_model
from sklearn.linear_model import Ridge

from sklearn.linear_model import LinearRegression
#lin_model=LinearRegression()
clf=linear_model.Lasso(alpha=0.1)
rid=Ridge(alpha=1.0)
import numpy as np
x1,x2,x3,y=df['Sale_1day'],df['Sale_2day'],df['Sale_3day'],df['New_cases']
x1,x2,x3,y=np.array(x1),np.array(x2),np.array(x3),np.array(y)
x1,x2,x3,y=x1.reshape(-1,1),x2.reshape(-1,1),x3.reshape(-1,1),y.reshape(-1,1)
final_x=np.concatenate((x1,x2,x3),axis=1)
#print(final_x)
X_train,X_test,y_train,y_test=final_x[:-30],final_x[-30:],y[:-30],y[-30:]
clf.fit(X_train,y_train)
rid.fit(X_train,y_train)
#lin_model.fit(X_train,y_train)
clf_pred=clf.predict(X_test)
rid_pred=rid.predict(X_test)

X_val_meta = np.column_stack((clf_pred,rid_pred))

# Train the meta-model on the combined feature matrix and the target values
#meta_model = LinearRegression()
#meta_model.fit(X_val_meta, y_test)
from sklearn import datasets, ensemble
params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}

meta_model = ensemble.GradientBoostingRegressor(**params)
meta_model.fit(X_val_meta, y_test)





print(y_test)




# Combine the predictions of the base models into a single feature matrix
X_new_meta = np.column_stack((clf_pred,rid_pred))

# Make a prediction using the meta-model
y_new_pred = meta_model.predict(X_new_meta)

#print("Predicted median value of owner-occupied homes: ${:.2f} thousand".format(y_new_pred[0]))

#lin_pred=lin_model.fit(X_test)
import matplotlib.pyplot as plt
#plt.rcParams["figure.figsize"] = (11,6)
#plt.plot(data['Date_reported'][len(data)-30:],y_new_pred,label='Lasso-Ridge-Predictions')
#plt.plot(data['Date_reported'][len(data)-30:],y_test,label='Actual Sales')
#plt.legend(loc="upper left")
#plt.show()
fig, ax = plt.subplots(figsize = (12,8))
ax.grid()
plt.plot(data['Date_reported'][len(data)-30:],y_new_pred)
plt.plot(data['Date_reported'][len(data)-30:],y_test)
plt.setp(ax.get_xticklabels(), rotation = 75, 
         ha = 'right', fontsize = 12, 
         color = 'blue')
plt.xlabel('Date reported')
plt.ylabel('Covid_cases')
plt.show()

from sklearn.metrics import mean_squared_error
from math import sqrt
m1=max(y_test)
m2=min(y_test)
n=sqrt(mean_squared_error(y_test,y_new_pred))
n=n/(m1-m2)
print(n)

from sklearn.metrics import mean_absolute_error
p=mean_absolute_error(y_test,y_new_pred)
p=p/(m1-m2)
print(p)

