import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
# Reading the data from panda dataframe
data = pd.read_csv("IPL 2022 Batters.csv")
# Checking for NULL value
data = data.fillna(0)
# Graph of runs vs 4s
plt.scatter(data["Runs"], data["4s"])
plt.xlabel("Runs")
plt.ylabel("Number of 4s")
plt.title("Runs vs Number of 4s")
plt.show()
# graph of SR vs 4s
plt.scatter(data["SR"], data["4s"])
plt.xlabel("SR")
plt.ylabel("Number of 4s")
plt.title("SR vs Number of 4s")
plt.show()

# Linear regression
lr = LinearRegression()
# features

features = data[["Runs", "SR"]]
# target
target = data["4s"]

# train_test_split utility from sklearn into training and testing with the number of 4s as a target.

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)
# finding mean squared value
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print("linear Train data MSE:", train_mse)
print("linear Test data MSE:", test_mse)
# doing logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_train_pred = logreg.predict(X_train)
y_test_pred = logreg.predict(X_test)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Logistic Train MSE:", train_mse)
print("Logistic Test MSE:", test_mse)
