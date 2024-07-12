import csv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import urllib

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)

def saham():
    time_step = []
    terakhir = []

    with open('Data_bersih.csv') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      next(reader)
      for row in reader:
        terakhir.append(float(row[2]))
        time_step.append(int(row[0]))

    series= np.array(terakhir)

    # Normalization Function. DO NOT CHANGE THIS CODE
    min=np.min(series)
    max=np.max(series)
    series -= min
    series /= max
    time=np.array(time_step)

    # DO NOT CHANGE THIS CODE
    split_time=3000


    time_train= time[:split_time]
    x_train= series[:split_time]
    time_valid= time[split_time:]
    x_valid= series[split_time:]

    # DO NOT CHANGE THIS CODE
    window_size=30
    batch_size=32
    shuffle_buffer_size=1000


    train_set=windowed_dataset(x_train, window_size=window_size,
                               batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)

    val_set = windowed_dataset(x_valid,window_size=window_size,
                               batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)

    model=tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=64,kernel_size=3,strides=1,activation='relu',padding='causal',input_shape=[None,1]),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # YOUR CODE
    model.compile(loss='mae',optimizer='Adam',metrics=['mae'])
    model.fit(train_set,epochs=100,validation_data=val_set,verbose=1)

    def model_forecast(model, series, window_size):
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window_size))
        ds = ds.batch(32).prefetch(1)
        forecast = model.predict(ds)
        return forecast

    def plot_series(time, series, format="-", start=0, end=None):
        plt.plot(time[start:end], series[start:end], format)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True)

    # Compute the forecast for all the series
    stsp_forecast = model_forecast(model, series, window_size).squeeze()

    # Slice the forecast to get only the predictions for the validation set
    stsp_forecast = stsp_forecast[split_time - window_size:-1]

    def compute_metrics(true_series, forecast):
        mse = tf.keras.metrics.mean_squared_error(true_series, forecast).numpy()
        mae = tf.keras.metrics.mean_absolute_error(true_series, forecast).numpy()
        mape = tf.keras.metrics.mean_absolute_percentage_error(true_series, forecast).numpy()
        return mse, mae, mape

    mse, mae, mape = compute_metrics(x_valid, stsp_forecast)
    print(f"mse: {mse:.5f}, mae: {mae:.5f}, mape: {mape:.5f}")

    # Plot the forecast
    plt.figure(figsize=(5, 3))
    plot_series(time_valid, x_valid)
    plot_series(time_valid, stsp_forecast)
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=saham()
    model.save("model_A5.h5")
