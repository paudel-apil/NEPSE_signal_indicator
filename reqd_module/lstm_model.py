import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_lstm_data(df, seq_length=10):
    data = df[['open_price', 'high_price', 'low_price', 'volume', 'turnover', 'close_price']]
    ss = StandardScaler()
    data = ss.fit_transform(data)
    X, y = create_sequences(data, seq_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test, ss

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length - 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length + 1, 0])
    return np.array(X), np.array(y)

def build_and_train_lstm_model(X_train, y_train, X_test, y_test):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=False, input_shape=(10, X_train.shape[2])),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.0024271328536722074), metrics=['mae'])
    model.fit(X_train, y_train, epochs=25, verbose=1, validation_data=(X_test, y_test), batch_size=32)
    return model

def predict_lstm_scores(df, model):
    data = df[['open_price', 'high_price', 'low_price', 'volume', 'turnover', 'close_price']].values
    seq_length = 10
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
    X = np.array(X)
    # Predict for the last 20% of the data (test set size)
    test_size = int(0.2 * len(df))
    start_idx = len(X) - test_size
    X_test_seq = X[start_idx:]
    preds = model.predict(X_test_seq)
    preds = preds.ravel()
    y_test = df['close_price'].shift(-1).iloc[-len(preds):].values
    res = pd.DataFrame({'Prediction': preds, 'True': y_test})
    res['error'] = np.abs(res['Prediction'] - res['True'])
    def score_fn(err):
        if err < 0.01:
            return 3
        elif err < 0.03:
            return 2
        elif err < 0.05:
            return 1
        else:
            return 0
    res['deep_score'] = res['error'].apply(score_fn)
    return res['deep_score'].values

#Used keras-tuner for finding the optimal parameters for the lstm model

'''
import keras_tuner
import keras


def build_model(hp):
  model = keras.Sequential()

  model.add(
        keras.layers.LSTM(
            units=hp.Int('units_1', min_value=32, max_value=128, step=16),
            return_sequences=True,
            input_shape=(seq_length, X_train.shape[2])
        )
    )
  model.add(keras.layers.Dropout(hp.Float('dropout_1', 0.1, 0.5, step=0.1)))

    # Optionally add a second LSTM layer
  if hp.Boolean('second_lstm'):
        model.add(
            keras.layers.LSTM(
                units=hp.Int('units_2', min_value=32, max_value=128, step=16),
                return_sequences=False
            )
        )
        model.add(keras.layers.Dropout(hp.Float('dropout_2', 0.1, 0.5, step=0.1)))
  else:
        model.add(keras.layers.LSTM(units=hp.Int('units_2', 32, 128, step=16), return_sequences=False))

  model.add(keras.layers.Dense(hp.Int('dense_units', 16, 64, step=16), activation='relu'))
  model.add(keras.layers.Dense(1))

  model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        ),
        loss='mse',
        metrics=['mse']
    )
  return model

from kerastuner.tuners import RandomSearch

tuner = RandomSearch(
    build_model,
    objective='val_mse',
    max_trials=20,        # number of different hyperparameter combinations to try
    executions_per_trial=2,  # number of model builds per combination (averaging)
    directory='kt_dir',
    project_name='nepse_lstm'
)

tuner.search(X_train, y_train,
             epochs=25,
             validation_data=(X_test, y_test),
             batch_size=32)


best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]


'''