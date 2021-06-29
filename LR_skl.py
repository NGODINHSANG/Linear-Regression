from sklearn import linear_model, datasets

import numpy as np
import pandas as pd

data = pd.read_csv("CN.txt")
data = data[["hight", "weight"]]

X = np.array(data[["hight"]])
y = np.array(data["weight"])

model = linear_model.LinearRegression()

model.fit(X, y)

print("coef : ", model.coef_)
print("Intercept : ", model.intercept_)
