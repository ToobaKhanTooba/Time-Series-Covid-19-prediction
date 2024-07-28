import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv(r'C:\Users\HP\OneDrive\Documents\research_theory\SARCOV.csv',parse_dates=True)
from statsmodels.tsa.arima_model import ARIMA

# First 30 days
#first_15 =df[:600]
#second_15=df[627:]
# fit model
import statsmodels.api as sm
#model = sm.tsa.arima.ARIMA(first_15.New_cases, order=(1,1,2))
#result = model.fit()
data = df['New_cases'].values
train_size = int(len(data))
train, test = data[0:train_size-30], data[train_size-30:]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = sm.tsa.arima.ARIMA(history, order=(1,1,0))
    model_fit = model.fit()
    pred = model_fit.forecast()
    yhat = pred[0]
    predictions.append(yhat)
    # Append test observation into overall record
    obs = test[t]
    history.append(obs)
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test, predictions))

print('Test RMSE:', rmse)

m1=max(test)
m2=min(test)
diff=m1-m2
nrmse=rmse/diff
print('nrmse',nrmse)


#print(test[:30])
#print(predictions[:30])

plt.plot(predictions)
plt.plot(test)

plt.legend(["predicted", "actual"], loc ="lower right")
plt.ylabel('confirmed cases')
plt.xlabel('Date')
plt.show()

#def mean_absolute_error(true, pred):
 #   abs_error = ((np.abs(true - pred))**2)/30
  #  return abs_error
#abs=mean_absolute_error(test,predictions)
#plt.plot(Date_reported[train_size-30:],abs)
#plt.ylabel('loss;')
#plt.xlabel('date')
#plt.show()

from sklearn.metrics import mean_absolute_error
p=mean_absolute_error(test,predictions)
p=p/(m1-m2)
print('mae',p)



