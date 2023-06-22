# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# from keras.models import Sequential
# from keras.layers import Dense, LSTM

# Load the data
df = pd.read_csv('./dataset/supermarket_sales - Sheet1.csv', parse_dates=['Date'])

df_base = df[['Date', 'Quantity']].copy()
plt.show()
df_base = df_base.groupby(['Date'], as_index=False)['Quantity'].sum()

print(df.columns)
print(df_base)
print(df[['Date', 'Quantity']])
exit()
df['Date'] = pd.to_datetime(df['Date'])

# Scale the 'DailyQty' column using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df['Scaled_DailyQty'] = scaler.fit_transform(np.array(df['DailyQty']).reshape(-1,1))

# Function to convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        X.append(a)
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)

# Using the function to prepare the data
look_back = 1
X, Y = create_dataset(df['Scaled_DailyQty'], look_back)

# Reshaping the input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# Splitting the data into training and testing sets
train_size = int(len(X) * 0.67)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size,:], X[train_size:len(X),:]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

# Creating and fitting the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2)

# Making predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverting predictions
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

# Calculating root mean squared error
train_score = np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0]))
print('Train Score: %.2f RMSE' % (train_score))
test_score = np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0]))
print('Test Score: %.2f RMSE' % (test_score))
