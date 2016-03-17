# use pandas to handle data
# use sklearn for machine-learning algos
import numpy as np
import random
import math
# Import the linear regression class
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas

sp500 = pandas.read_csv("sp500.csv")
sp500.head()

# clean data
sp500 = sp500[sp500["value"] != "."]
# finding predictors
next_day = sp500["value"].iloc[1:]
sp500 = sp500.iloc[:-1,:]
sp500["next_day"] = next_day.values
# convert columns to floats
print(sp500.dtypes)
sp500["value"] = sp500["value"].astype(float)
sp500["next_day"] = sp500["next_day"].astype(float)

# Set a random seed to make the shuffle deterministic.
np.random.seed(1)
random.seed(1)
# Randomly shuffle the rows in our dataframe
sp500 = sp500.loc[np.random.permutation(sp500.index)]

# Select 70% of the dataset to be training data
highest_train_row = int(sp500.shape[0] * .7)
train = sp500.loc[:highest_train_row,:]

# Select 30% of the dataset to be test data.
test = sp500.loc[highest_train_row:,:]

# Initialize the linear regression class.
regressor = LinearRegression()
# We pass in a list when we select predictor columns from "sp500" to force pandas not to generate a series.
# Train the linear regression model on our dataset.
regressor.fit(train[["value"]], train["next_day"])
# Generate a list of predictions with our trained linear regression model
predictions = regressor.predict(test[["value"]])
mse = sum((predictions - test["next_day"]) ** 2) / len(predictions)

plt.scatter(test["value"], test["next_day"])
plt.plot(test["value"], predictions)
plt.show()

# other error metrics
rmse = math.sqrt(sum((predictions - test["next_day"]) ** 2) / len(predictions))
mae = sum(abs(predictions - test["next_day"])) / len(predictions)
