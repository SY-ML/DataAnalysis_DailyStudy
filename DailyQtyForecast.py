from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd

class RunLSTM():
    def __init__(self, df, col_indVar, col_depVar, col_time):
        self._df = df
        self._col_iv = col_indVar
        self._col_dv = col_depVar if isinstance(col_depVar, list) else [col_depVar]
        self._col_t = col_time

    def preprocess_dataset(self):
        df = self._df.copy()
        df[self._col_t] = pd.to_datetime(df[self._col_t])
        return df

    def create_model(self, look_back, optimizer='adam'):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.summary()
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    def create_dataset(self, dataset, look_back):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    def predict_next_day(self, look_back, epochs=50, batch_size=10):
        df = self.preprocess_dataset()

        dataset = df.values
        dataset = dataset.astype('float32')

        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

        X_train, y_train = self.create_dataset(train, look_back)
        X_test, y_test = self.create_dataset(test, look_back)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        model = self.create_model(look_back, optimizer=Adam())

        checkpoint_callback = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(X_train, y_train,
                            validation_data=(X_test, y_test),
                            callbacks=[early_stopping_callback, checkpoint_callback],
                            epochs=epochs, batch_size=batch_size)

        return model


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
rl = RunLSTM(df=df, col_time='Date', col_indVar='Qty(prev)', col_depVar='Qty')
print(rl.preprocess_dataset())
rl.predict_next_day(look_back=1)
exit()
