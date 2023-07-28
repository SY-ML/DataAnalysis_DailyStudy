from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import csv
import os

class RunLSTM():
    def __init__(self, df, col_indVar, col_depVar, col_time):
        self._df = df
        self._col_iv = col_indVar if isinstance(col_indVar, list) else [col_indVar]
        self._col_dv = col_depVar if isinstance(col_depVar, list) else [col_depVar]
        self._col_t = col_time

    def preprocess_dataset(self):
        df = self._df.copy()

        # String to date
        df[self._col_t] = pd.to_datetime(df[self._col_t])

        # drop
        df.dropna(inplace=True)

        return df

    def log_parameters(self, history, optimizer, look_back, epochs, batch_size):
        layer1 = self.lstm.layers[0]
        layer2 = self.lstm.layers[1]

        log_data = {
            'Epochs': epochs,
            'Batch Size': batch_size,
            'Lookback': look_back,
            'Layer1 Model': layer1.__class__.__name__,
            'Layer1 Params': layer1.count_params(),
            'Layer2 Model': layer2.__class__.__name__,
            'Layer2 Params': layer2.count_params(),
            'Optimizer': optimizer,
            'Final Training Loss': history.history['loss'][-1],
            'Final Validation Loss': history.history['val_loss'][-1],
        }

        # Convert to DataFrame
        df_log = pd.DataFrame([log_data])

        # Append to the log file (or create it if it doesn't exist)
        df_log.to_csv('./dataset/supermarket_sales - Sheet1_testlog.csv', mode='a',
                      header=not os.path.exists('./dataset/supermarket_sales - Sheet1_testlog.csv'), index=False)


    def create_model(self, look_back, num_features, optimizer='adam'):
        model = Sequential()
        model.add(LSTM(50, activation='tanh', return_sequences=True, input_shape=(look_back, num_features)))
        model.add(Dense(1))
        model.summary()
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['MeanSquaredError'])
        return model

    def predict_next_day(self, look_back, epochs=50, batch_size=10):
        df = self.preprocess_dataset()

        inputs = df[self._col_iv].values
        target = df[self._col_dv].values

        loss_history = []  # This list will store the loss of each training iteration

        # Create the LSTM model outside the loop
        lstm = self.create_model(look_back=look_back, optimizer='Adam', num_features=1)

        for i in range(look_back, len(inputs)):
            data_input = inputs[:i]
            data_target = target[:i]

            scaler_train = StandardScaler()
            scaler_target = StandardScaler()

            data_std_train = scaler_train.fit_transform(data_input.reshape(-1, 1))
            data_std_target = scaler_target.fit_transform(data_target.reshape(-1, 1))

            print(data_std_train)
            print(data_std_target)

            # Create datasets
            checkpoint_callback = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='loss', mode='min')
            early_stopping_callback = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            # checkpoint_callback = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
            # early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            # Calculate the validation split ratio for the last row
            validation_split_ratio = min(1 / data_std_train.shape[0], 0.2)  # limit to 20% of the data


            # Fit the model to the training data
            # Fit the model to the training data
            history = lstm.fit(data_std_train, data_std_target,
                               callbacks=[early_stopping_callback, checkpoint_callback],
                               epochs=epochs, batch_size=batch_size,
                               validation_split=min(1 / data_std_train.shape[0], 0.2))

            # Append the final loss of this iteration to the list
            loss_history.append(history.history['loss'][-1])
            print(f"Iteration {i}, Loss: {history.history['loss'][-1]}")

        # At this point, loss_history contains the final loss of each training iteration.
        # You can return it or use it directly to plot your results.
        return loss_history


def load_mydataset():
    df = pd.read_csv('./dataset/supermarket_sales - Sheet1.csv', parse_dates=['Date'])
    df = df.groupby(['Date'], as_index=True)['Quantity'].sum().reset_index()

    # Rename 'Quantity' to 'Qty(prev)'
    df.rename(columns={'Quantity': 'Qty(prev)'}, inplace=True)

    # Then create 'Qty' based on the shifted values of 'Qty(prev)'
    df['Qty'] = df['Qty(prev)'].shift(-1)

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
for int in [3, 5, 10, 30, 60]:
    df[f'Qty(ma-{int})'] = df['Qty(prev)'].rolling(int).mean()

rl = RunLSTM(df=df, col_time='Date', col_indVar='Qty(prev)', col_depVar='Qty')


# Call the method and get the loss history
loss_history = rl.predict_next_day(5)

# Plot the loss history
plt.plot(loss_history)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.show()
