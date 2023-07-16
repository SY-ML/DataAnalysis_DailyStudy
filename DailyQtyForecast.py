from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


class RunLSTM():
    def __init__(self, df, col_indVar, col_depVar, col_time):
        self._df = df
        self._col_iv = col_indVar if isinstance(col_indVar, list) else [col_indVar]
        self._col_dv = col_depVar if isinstance(col_depVar, list) else [col_depVar]
        self._col_t = col_time
        self.predict_next_day(1)

    def preprocess_dataset(self):
        df = self._df.copy()

        # String to date
        df[self._col_t] = pd.to_datetime(df[self._col_t])

        # drop
        df.dropna(inplace=True)

        return df

    def create_model(self, look_back, num_features, optimizer='adam'):
        model = Sequential()
        model.add(LSTM(50, activation='tanh', return_sequences=True, input_shape=(look_back, num_features)))
        model.add(Dense(1))
        model.summary()
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['MeanSquaredError'])
        return model

    def predict_next_day(self, epochs=50, batch_size=10):
        df = self.preprocess_dataset()

        inputs = df[self._col_iv].values
        target = df[self._col_dv].values

        for i in range(1, len(inputs)):
            data_input = inputs[:i].reshape(-1, 1)
            data_target = target[:i].reshape(-1, 1)

            scaler_train = StandardScaler()
            scaler_target = StandardScaler()

            data_std_train = scaler_train.fit_transform(data_input)
            data_std_target = scaler_target.fit_transform(data_target)

            # Create the LSTM model
            lstm = self.create_model(look_back=1, optimizer='Adam', num_features=data_std_train.shape[1])

            # Create datasets
            # checkpoint_callback = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='loss', mode='min')
            # early_stopping_callback = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            # lstm.fit(data_std_train, data_std_target, callbacks=[early_stopping_callback, checkpoint_callback],

            # Fit the model to the training data
            lstm.fit(data_std_train, data_std_target, epochs=epochs, batch_size=batch_size)


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
