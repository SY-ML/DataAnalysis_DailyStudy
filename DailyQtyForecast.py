import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import csv
import os


class RunLSTM():
    def __init__(self, df, col_indVar, col_depVar, col_time):
        """
        Initialize the LSTM model.

        Args:
            df: A Pandas DataFrame that contains the dataset.
            col_indVar: A string or list of strings that represent the column name(s) of the independent variable(s).
            col_depVar: A string or list of strings that represent the column name(s) of the dependent variable(s).
            col_time: A string that represents the column name of the time data.
        """
        self._df = df
        self._col_iv = col_indVar
        self._col_dv = col_depVar if isinstance(col_depVar, list) else [col_depVar]
        self._col_t = col_time

    def preprocess_dataset(self):
        """
        Preprocesses the dataset: groups by date, fills missing dates with 0, shifts the quantities to create
        new 'Qty (yesterday)' columns for each dependent variable.

        Returns:
            dataframe: The preprocessed dataframe.
        """
        # Copy the dataframe to avoid modifying the original
        df = self._df.copy()

        # Parse the dates
        df[self._col_t] = pd.to_datetime(df[self._col_t])

        return df
def load_mydataset():
    df = pd.read_csv('./dataset/supermarket_sales - Sheet1.csv', parse_dates=['Date'])
    df = df.groupby(['Date'], as_index=True)['Quantity'].sum().reset_index()

    # Rename 'Quantity' to 'Qty(prev)'
    df.rename(columns={'Quantity': 'Qty(prev)'}, inplace=True)

    # Then create 'Qty' based on the shifted values of 'Qty(prev)'
    df['Qty'] = df['Qty(prev)'].shift(1)

    # Handle NaN values and convert the column back to integer
    df['Qty'] = df['Qty'].fillna(0).astype(int)

    # Get the min and max dates
    start_date = df['Date'].min()
    end_date = df['Date'].max()

    # Create a date range
    date_range = pd.date_range(start=start_date, end=end_date)

    # Set the date as the index
    df.set_index('Date', inplace=True)

    # Reindex the dataframe
    df = df.reindex(date_range, fill_value=0)

    # Reset the index so 'Date' becomes a column again
    df.reset_index(inplace=True)

    # Rename the 'index' column to 'Date'
    df.rename(columns={'index': 'Date'}, inplace=True)

    return df

df = load_mydataset()
print(df)
rl = RunLSTM(df = df, col_time='Date', col_indVar='Qty(prev)', col_depVar='Qty')
print(rl.preprocess_dataset())
exit()
#
# def create_dataset(dataset, look_back=1):
#     X, Y = [], []
#     for i in range(len(dataset) - look_back - 1):
#         a = dataset[i:(i + look_back)]
#         X.append(a)
#         Y.append(dataset[i + look_back])
#     return np.array(X), np.array(Y)
#
#
# def create_model(look_back, optimizer='adam'):
#     model = Sequential()
#     model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
#     model.add(LSTM(50))
#     model.add(Dense(1))
#     model.summary()
#     model.compile(loss='mean_squared_error', optimizer=optimizer)
#     return model
#
#
# def predict_next_day(df, look_back, epochs=50, batch_size=10):
#     df, scaler = preprocess_dataset(df=df)
#
#     train_size = int(len(df) * 0.8)
#     test_size = len(df) - train_size
#     train, test = df[0:train_size, :], df[train_size:len(df), :]
#
#     X_train, y_train = create_dataset(train, look_back)
#     X_test, y_test = create_dataset(test, look_back)
#
#     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#
#     model = create_model(look_back, optimizer=Adam())
#
#     checkpoint_callback = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
#     early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#
#     history = model.fit(X_train, y_train,
#                         validation_data=(X_test, y_test),
#                         callbacks=[early_stopping_callback, checkpoint_callback],
#                         epochs=epochs, batch_size=batch_size)
#
#     return model, scaler
#
#
# # Load the data
# df = pd.read_csv('./dataset/supermarket_sales - Sheet1.csv', parse_dates=['Date'])
# df_base = df[['Date', 'Quantity']].copy()
#
# # Train the model and get the scaler
# model, scaler = predict_next_day(df_base, look_back=1)
