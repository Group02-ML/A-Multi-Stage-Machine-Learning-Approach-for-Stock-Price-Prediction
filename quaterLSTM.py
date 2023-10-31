#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_excel('JKH 2022-2023.xlsx', parse_dates=['Day'])
#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

for i in range(0,len(data)):
    new_data['Date'][i] = data['Day'][i]
    new_data['Close'][i] = data['Close'][i]

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)
print(new_data)
#creating train and test sets
dataset = new_data.values
# split data set quarterly
train = dataset[0:193,:]
valid = dataset[193:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=2)

#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

#for plotting
train = new_data[:193]
valid = new_data[193:]
#valid['Predictions'] = closing_price
#valid.loc[:, 'Predictions'] = closing_price
valid = valid.copy()
valid['Predictions'] = closing_price
plt.plot(train['Close'],label='Training Data',color='g')
plt.plot(valid[['Close']],label='Testing Data',color='b')
plt.plot(valid[['Predictions']],label='Predicted stock price',color='r')
plt.title("Quarterly Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Stock Price Value")
plt.legend(loc='best')
plt.show()
valid = valid.copy()
valid['Predictions'] = closing_price
plt.plot(valid[['Close']],label='Actual stock price',color='b')
plt.plot(valid[['Predictions']],label='Predicted stock price',color='r')
plt.title("Quarterly Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Stock Price Value")
plt.legend(loc='best')
plt.show()