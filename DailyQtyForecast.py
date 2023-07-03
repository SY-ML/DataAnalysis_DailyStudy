import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.callbacks import EarlyStopping
import numpy as np

print("GPU Available: ", tf.config.list_physical_devices('GPU'))
print("CUDA Version: ", tf.sysconfig.get_build_info()["cuda_version"])
print("cuDNN Version: ", tf.sysconfig.get_build_info()["cudnn_version"])
exit()
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        X.append(a)
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)

df = pd.read_csv('./dataset/supermarket_sales - Sheet1.csv', parse_dates=['Date'])
df_base = df[['Date', 'Quantity']].copy()

def preprocess_dataset(df):
    # Convert the 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Group by 'Date' and sum 'Quantity' to ensure unique dates
    df = df.groupby('Date')['Quantity'].sum().reset_index()

    # Find the earliest date (start_date)
    start_date = df['Date'].min()

    # Find the latest date (end_date)
    end_date = df['Date'].max()

    print('Start Date:', start_date)
    print('End Date:', end_date)

    # Create a date range from start to end
    date_range = pd.date_range(start=start_date, end=end_date)

    # Set 'Date' as the index of df_base for the reindex operation
    df.set_index('Date', inplace=True)

    # Reindex the dataframe to the complete date range
    df = df.reindex(date_range, fill_value=0)

    # Reset the index to bring 'Date' back as a column
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Date'}, inplace=True)

    # Generate shifted 'Qty (yesterday)' column
    df['Qty (yesterday)'] = df['Quantity'].shift(fill_value=0)

    # Rename 'Quantity' to 'Qty (today)'
    df.rename(columns={'Quantity': 'Qty (today)'}, inplace=True)

    return df

df_base = preprocess_dataset(df = df_base)

def create_model(look_back, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def predict_next_day(df, look_back):
    # Normalize the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(np.array(df['Qty (today)']).reshape(-1, 1))

    # Split the dataset into training and testing sets
    train_size = int(len(df) * 0.8)  # using 80% of the data for training
    test_size = len(df) - train_size
    train, test = df[0:train_size,:], df[train_size:len(df),:]

    # Reshape data to be [samples, time steps, features]
    X_train, y_train = create_dataset(train, look_back)
    X_test, y_test = create_dataset(test, look_back)

    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Create the LSTM model
    model = KerasRegressor(build_fn=create_model, verbose=0, look_back=look_back)

    # Define the grid search parameters
    batch_size = [10, 20, 40, 60, 80, 100]
    epochs = [10, 50, 100]
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

    # Train the model
    grid_result = grid.fit(X_train, y_train,
                           validation_data=(X_test, y_test),
                           callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=0)

    # Summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

predict_next_day(df_base, look_back=1)


"""
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

print(df_base)

# Scale the 'DailyQty' column using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df_base['Scaled_DailyQty'] = scaler.fit_transform(np.array(df_base['DailyQty']).reshape(-1,1))

#062723 environment reinstalled
# Function to convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        X.append(a)
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)

print(df_base)
# Using the function to prepare the data
look_back = 1
X, Y = create_dataset(df_base['Scaled_DailyQty'], look_back)  # Change df to df_base here

# Reshaping the input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

print(X)
exit()

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
"""
