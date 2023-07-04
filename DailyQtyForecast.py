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
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.groupby('Date')['Quantity'].sum().reset_index()
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    date_range = pd.date_range(start=start_date, end=end_date)
    df.set_index('Date', inplace=True)
    df = df.reindex(date_range, fill_value=0)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Date'}, inplace=True)
    df['Qty (yesterday)'] = df['Quantity'].shift(fill_value=0)
    df.rename(columns={'Quantity': 'Qty (today)'}, inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(np.array(df['Qty (today)']).reshape(-1, 1))
    return df


df_base = preprocess_dataset(df=df_base)


def create_model(look_back, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.summary()
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def predict_next_day(df, look_back, epochs=50, batch_size=10):

    train_size = int(len(df) * 0.8)
    test_size = len(df) - train_size
    train, test = df[0:train_size,:], df[train_size:len(df),:]

    X_train, y_train = create_dataset(train, look_back)
    X_test, y_test = create_dataset(test, look_back)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = create_model(look_back, optimizer=Adam())

    checkpoint_callback = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10), checkpoint_callback],
                        epochs=epochs, batch_size=batch_size)

    visualize_performance(history)

    testlog_filename = './dataset/supermarket_sales - Sheet1_testlog.csv'
    write_testlog(testlog_filename, model, history, epochs, batch_size)

def visualize_performance(history):
    plt.figure(figsize=(10, 6))

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model performance')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

def write_testlog(filename, model, history, epochs, batch_size):
    fieldnames = ['Epochs', 'Batch Size', 'Layer1 Model', 'Layer1 Params', 'Layer2 Model', 'Layer2 Params', 'Optimizer', 'Final Training Loss', 'Final Validation Loss']

    layer1_model = type(model.layers[0]).__name__
    layer1_params = model.layers[0].count_params()

    layer2_model = type(model.layers[1]).__name__
    layer2_params = model.layers[1].count_params()

    optimizer = type(model.optimizer).__name__

    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]

    log_data = [epochs, batch_size, layer1_model, layer1_params, layer2_model, layer2_params, optimizer, final_train_loss, final_val_loss]
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({field: data for field, data in zip(fieldnames, log_data)})

predict_next_day(df_base, look_back=1)
