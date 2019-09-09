import pandas as pd
import tensorflow.python.keras as keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn
import sys
import tensorflow as tf
import time

temps = pd.read_csv(filepath_or_buffer='./data/daily-minimum-temperatures-in-me.csv', parse_dates=[0], index_col=0)
#print(type(temps))
#print(temps.info())
#print(temps.head())
#temps.plot(figsize=(10, 5))
#plt.show()

temps = temps.asfreq(freq='1D', method='ffill')


def add_lags(series, times):
    cols = []
    column_index = []
    for time in times:
        cols.append(series.shift(-time))
        lag_fmt = "t+{time}" if time > 0 else "t{time}" if time < 0 else "t"
        column_index += [(lag_fmt.format(time=time), col_name) for col_name in series.columns]
    df = pd.concat(cols, axis=1)
    df.columns = pd.MultiIndex.from_tuples(column_index)
    return df


X = add_lags(temps, times=range(-30+1, 1)).iloc[30:-5]
y = add_lags(temps, times=[5]).iloc[30:-5]

train_slice = slice(None, "1986-12-25")
valid_slice = slice("1987-01-01", "1988-12-25")
test_slice = slice("1989-01-01", None)

X_train, y_train = X.loc[train_slice], y.loc[train_slice]
X_valid, y_valid = X.loc[valid_slice], y.loc[valid_slice]
X_test, y_test = X.loc[test_slice], y.loc[test_slice]


def multilevel_df_to_ndarray(df):
    shape = [-1] + [len(level) for level in df.columns.remove_unused_levels().levels]
    return df.values.reshape(shape)


X_train_3D = multilevel_df_to_ndarray(X_train)
X_valid_3D = multilevel_df_to_ndarray(X_valid)
X_test_3D = multilevel_df_to_ndarray(X_test)

print(X_train.shape)
print(X_train_3D.shape)

#########################################
########  build baseline models  ########
#########################################

def plot_predictions(*named_predictions, start=None, end=None, **kwargs):
    day_range = slice(start, end)
    plt.figure(figsize=(10,5))
    for name, y_pred in named_predictions:
        if hasattr(y_pred, "values"):
            y_pred = y_pred.values
        plt.plot(y_pred[day_range], label=name, **kwargs)
    plt.legend()
    plt.show()

def plot_history(history, loss="loss"):
    train_losses = history.history[loss]
    valid_losses = history.history["val_" + loss]
    n_epochs = len(history.epoch)
    minloss = np.min(valid_losses)

    plt.plot(train_losses, color="b", label="Train")
    plt.plot(valid_losses, color="r", label="Validation")
    plt.plot([0, n_epochs], [minloss, minloss], "k--", label="Min val: {:.2f}".format(minloss))
    plt.axis([0, n_epochs, 0, 20])
    plt.legend()
    plt.show()

def huber_loss(y_true, y_pred, max_grad=1.):
    err = tf.abs(y_true - y_pred, name='abs')
    mg = tf.constant(max_grad, name='max_grad')
    lin = mg * (err - 0.5 * mg)
    quad = 0.5 * err * err
    return tf.where(err < mg, quad, lin)

K = keras.backend
def mae_last_step(Y_true, Y_pred):
    return K.mean(K.abs(Y_pred[:, -1] - Y_true[:, -1]))

from sklearn.metrics import mean_absolute_error


def naive(X):
    return X.iloc[:, -1]

y_pred_naive = naive(X_valid)
print('loss-1: ', mean_absolute_error(y_valid, y_pred_naive))


def ema(X, span):
    return X.T.ewm(span=span).mean().T.iloc[:, -1]

y_pred_ema = ema(X_valid, span=10)
print('loss-2: ', mean_absolute_error(y_valid, y_pred_ema))


from sklearn.linear_model import LinearRegression


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_linear = lin_reg.predict(X_valid)
print('loss-3: ', mean_absolute_error(y_valid, y_pred_linear))


plot_predictions(("Target", y_valid),
                 ("Naive", y_pred_naive),
                 ("EMA", y_pred_ema),
                 ("Linear", y_pred_linear),
                 end=365)

# simple RNN
# a simple 2-layer RNN with 100 neurons per layer, plus a dense layer with a single neuron
input_shape = X_train_3D.shape[1:]
model1 = keras.models.Sequential()
model1.add(keras.layers.SimpleRNN(100, return_sequences=True, input_shape=input_shape))
model1.add(keras.layers.SimpleRNN(50))
model1.add(keras.layers.Dense(1))
model1.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=0.005), metrics=["mae"])
history1 = model1.fit(X_train_3D, y_train, epochs=200, batch_size=200, validation_data=(X_valid_3D, y_valid))
plot_history(history1)
print('loss-4: ', model1.evaluate(X_valid_3D, y_valid)[1])
y_pred_rnn1 = model1.predict(X_valid_3D)

input_shape = X_train_3D.shape[1:]
Y = add_lags(temps, times=range(-24, 5+1)).iloc[30:-5]
Y_train = Y.loc[train_slice]
Y_valid = Y.loc[valid_slice]
Y_test = Y.loc[test_slice]
Y_train_3D = multilevel_df_to_ndarray(Y_train)
Y_valid_3D = multilevel_df_to_ndarray(Y_valid)
Y_test_3D = multilevel_df_to_ndarray(Y_test)

# For the final evaluation, we only want to look at the final time step (t+5):
model2 = keras.models.Sequential()
model2.add(keras.layers.SimpleRNN(100, return_sequences=True, input_shape=input_shape))
model2.add(keras.layers.SimpleRNN(100, return_sequences=True))
model2.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))
model2.compile(loss=huber_loss, optimizer=keras.optimizers.SGD(lr=0.01),
               metrics=[mae_last_step])
history2 = model2.fit(X_train_3D, Y_train_3D, epochs=200, batch_size=200,
                      validation_data=(X_valid_3D, Y_valid_3D))

plot_history(history2, loss="mae_last_step")
print('loss-5: ', model2.evaluate(X_valid_3D, Y_valid_3D)[1])
#y_pred_rnn2 = model2.predict(X_valid_3D)[:, -1]

Y = add_lags(temps, times=range(-24, 5+1)).iloc[30:-5]
Y_train = Y.loc[train_slice]
Y_valid = Y.loc[valid_slice]
Y_test = Y.loc[test_slice]
Y_train_3D = multilevel_df_to_ndarray(Y_train)
Y_valid_3D = multilevel_df_to_ndarray(Y_valid)
Y_test_3D = multilevel_df_to_ndarray(Y_test)
input_shape = X_train_3D.shape[1:]
model3 = keras.models.Sequential()
model3.add(keras.layers.LSTM(100, return_sequences=True, input_shape=input_shape))
model3.add(keras.layers.LSTM(100, return_sequences=True))
model3.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))
model3.compile(loss=huber_loss, optimizer=keras.optimizers.SGD(lr=0.01),
               metrics=[mae_last_step])
history3 = model3.fit(X_train_3D, Y_train_3D, epochs=200, batch_size=200,
                      validation_data=(X_valid_3D, Y_valid_3D),
                      callbacks=[keras.callbacks.ReduceLROnPlateau(verbose=1)])
plot_history(history3, loss="mae_last_step")
y_pred_rnn3 = model3.predict(X_valid_3D)[:, -1]
print('loss-3: ', model3.evaluate(X_valid_3D, Y_valid_3D)[1])


Y = add_lags(temps, times=range(-24, 5+1)).iloc[30:-5]
Y_train = Y.loc[train_slice]
Y_valid = Y.loc[valid_slice]
Y_test = Y.loc[test_slice]
Y_train_3D = multilevel_df_to_ndarray(Y_train)
Y_valid_3D = multilevel_df_to_ndarray(Y_valid)
Y_test_3D = multilevel_df_to_ndarray(Y_test)
input_shape = X_train_3D.shape[1:]
from functools import partial
RegularizedLSTM = partial(keras.layers.LSTM,
                          return_sequences=True,
                          kernel_regularizer=keras.regularizers.l2(1e-4),
                          recurrent_regularizer=keras.regularizers.l2(1e-4))
model4 = keras.models.Sequential()
model4.add(RegularizedLSTM(100, input_shape=input_shape))
model4.add(RegularizedLSTM(100))
model4.add(keras.layers.Dense(1))
model4.compile(loss=huber_loss, optimizer=keras.optimizers.SGD(lr=0.01),
               metrics=[mae_last_step])
history4 = model4.fit(X_train_3D, Y_train_3D, epochs=200, batch_size=100,
                      validation_data=(X_valid_3D, Y_valid_3D))
plot_history(history4)
y_pred_rnn4 = model4.predict(X_valid_3D)[:, -1]
print('loss-4: ', model4.evaluate(X_valid_3D, Y_valid_3D)[1])

Y = add_lags(temps, times=range(-24, 5+1)).iloc[30:-5]
Y_train = Y.loc[train_slice]
Y_valid = Y.loc[valid_slice]
Y_test = Y.loc[test_slice]
Y_train_3D = multilevel_df_to_ndarray(Y_train)
Y_valid_3D = multilevel_df_to_ndarray(Y_valid)
Y_test_3D = multilevel_df_to_ndarray(Y_test)
input_shape = X_train_3D.shape[1:]
model5 = keras.models.Sequential()
model5.add(keras.layers.Conv1D(32, kernel_size=5, input_shape=input_shape))
model5.add(keras.layers.MaxPool1D(pool_size=5, strides=2))
model5.add(keras.layers.LSTM(32, return_sequences=True))
model5.add(keras.layers.LSTM(32))
model5.add(keras.layers.Dense(1))
model5.compile(loss=huber_loss, optimizer=keras.optimizers.SGD(lr=0.005))

history5 = model5.fit(X_train_3D, y_train, epochs=200, batch_size=100,
                      validation_data=(X_valid_3D, y_valid))
plot_history(history5)
print('loss-5: ', model5.evaluate(X_valid_3D, Y_valid_3D))
