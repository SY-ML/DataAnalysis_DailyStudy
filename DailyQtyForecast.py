import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('data.csv')

# convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Make sure your data is sorted by date
data.sort_values('Date', inplace=True, ascending=True)

# Use 'Date' as index
data.set_index('Date', inplace=True)

# Select 'DailyQty' as the target variable
target_var = 'DailyQty'

# Normalize the features to be between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Split data into training and test sets. Let's use 70% for training and 30% for testing
train_size = int(len(scaled_data) * 0.7)
train, test = scaled_data[:train_size, :], scaled_data[train_size:, :]

# Convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i+look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, data.columns.get_loc(target_var)])
    return np.array(X), np.array(Y)

look_back = 1
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

# LSTM expects the input data in the form: [samples, time steps, features]
# So, we reshape the input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

# make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# invert predictions to original scale for interpretation
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# plot actual data vs predictions
plt.plot(scaler.inverse_transform(scaled_data[data.columns.get_loc(target_var)]), color='b')
plt.plot(np.concatenate((train_predict,test_predict)), color='r')
plt.show()
