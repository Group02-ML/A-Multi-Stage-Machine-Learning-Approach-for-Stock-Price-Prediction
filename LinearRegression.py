import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

data = pd.read_excel('Harischandra.xlsx', parse_dates=['Day'])
# Check data types in columns
print(data.info())
df_Stock = data
df_Stock.head()
print(df_Stock)
df_Stock.tail(5)
df_Stock.set_index('Day', inplace=True)
# High_range = df_Stock['High']
# Low_range = df_Stock['Low']
Close_range = df_Stock['Closing']
# Create a new DataFrame with only closing price and date
df = pd.DataFrame(data, columns=['Closing'])

# Reset index column so that we have integers to represent time for later analysis
df = df.reset_index()
print(df)
df.isna().values.any()
# Import package for splitting data set
from sklearn.model_selection import train_test_split

# Split data into train and test set: 80% / 20%
train, test = train_test_split(df, test_size=0.05, random_state=42)
# Import package for linear model
from sklearn.linear_model import LinearRegression

# Reshape index column to 2D array for .fit() method
X_train = np.array(train.index).reshape(-1, 1)
y_train = train['Closing']

# Create LinearRegression Object
model = LinearRegression()
# Fit linear model using the train data set
model.fit(X_train, y_train)
# The coefficient
slope = model.coef_.item()
print('Slope:', slope)
slope = model.coef_[0]
print('Slope:', slope)
# The Intercept
print('Intercept: ', model.intercept_)
# Create test arrays
X_test = np.array(test.index).reshape(-1, 1)
y_test = test['Closing']
# Generate array with predicted values
y_pred = model.predict(X_test)


def get_mape(y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Calculate and print values of MAE, MSE, RMSE
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred), 2))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('\n\nMean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred), 6))
print("R-squared error: ", round(metrics.r2_score(y_test, y_pred), 6))
print('Validation MAPE:', round(get_mape(y_test, y_pred), 6))