import pandas as pd
import quandl
import math, datetime
import numpy as np
import sklearn.model_selection
from sklearn import preprocessing, svm, linear_model
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get("WIKI/GOOGL")

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Open']  * 100
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

forescast_col = 'Adj. Close'
df.fillna(-9999, inplace=True)

forescast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forescast_col].shift(-forescast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forescast_out:]
X = X[:-forescast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])
y = y[:-forescast_out:]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

#model = linear_model.LinearRegression(n_jobs=-1)
#model.fit(X_train, y_train)
#with open('linearregression.pickle', 'wb') as f:
#    pickle.dump(model, f)

pickle_in = open('linearregression.pickle', 'rb')
model = pickle.load(pickle_in)

acc = model.score(X_test, y_test)

forescast_set = model.predict(X_lately)

print(forescast_set, acc, forescast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forescast_set:
    next_day = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_day] = [np.nan for _ in range(len(df.columns)-1)] +[i]

print(df.head())
print(df.tail())
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


