import numpy as np
import pandas as pd

data = pd.read_csv("CN.txt")

data = data[["hight", "weight"]]

X = np.array(data[["hight"]])

y = np.array(data["weight"])

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

A=np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)

w0, w1 = w[0], w[1]

y1 = w1*X[1][0] + w0
y2 = w1*X[6][0] + w0

print('Input '+str(X[1][0]) +'cm, true output ' + str(y[1]) + ' predicted output %.2fkg' %(y1))
print('Input '+str(X[6][0]) +'cm, true output ' + str(y[6]) + ' predicted output %.2fkg' %(y2))

print(w1)
print(w0)