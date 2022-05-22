import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
# from pyodide.http import pyfetch
# %matplotlib inline

# path =  "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

# async def download(url, filename):
#     response = await pyfetch(url)
#     if response.status == 200:
#         with open(filename, "wb") as f:
#             f.write(await response.bytes())
#     await download(path, "FuelConsumptionCo2.csv")
#     path="FuelConsumptionCo2.csv"


# read data
data = pd.read_csv("./FuelConsumptionCo2.csv")


# visualize data
data.head()
data.describe()
data.hist()
data.shape

cdf = data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

# scatter plot  fuelconsumption vs emission
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

# scatter plot enginesize vs emission
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# scatter plot of cylinder vs emission
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show()

# split data 80% for train and 20% for test
msk = np.random.rand(len(data)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# scatter plot of train dataset (enginesize vs emisson)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# apply linear regression to get the coefficient and intercept
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])

train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
# print('train_x:' ,train_x[0][0])

# plot and set label
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

# get accuracy and mae(mean absolute error)
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_new = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_new - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_new - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_new) )