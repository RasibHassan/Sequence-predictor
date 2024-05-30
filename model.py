import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_data(filepath):
    lottery_data = pd.read_csv(filepath)
    lottery_numbers = lottery_data[['First Number', 'Second Number', 'Third Number', 'Fourth Number', 'Fifth Number']].values
    lottery_numbers_normalized = (lottery_numbers - 1) / 35
    return lottery_numbers_normalized

def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        xs.append(data[i:(i + sequence_length)])
        ys.append(data[i + sequence_length])
    return np.array(xs), np.array(ys)

def adjust_initial_biases(all_numbers):
    number_freq = pd.Series(all_numbers).value_counts(normalize=True)
    initial_biases = np.log(number_freq.sort_index().values + 1e-5)
    return initial_biases

def lstm_model(input_shape, biases):
    inputs = tf.keras.Input(shape=input_shape)
    lstm_layer1 = tf.keras.layers.LSTM(64, return_sequences=True, activation='tanh', dropout=0.2)(inputs)
    lstm_layer2 = tf.keras.layers.LSTM(32, return_sequences=False, activation='tanh', dropout=0.2)(lstm_layer1)
    predictions = [tf.keras.layers.Dense(36, activation='softmax', bias_initializer=tf.keras.initializers.Constant(biases))(lstm_layer2) for _ in range(5)]
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def get_unique_numbers(predictions):
    chosen_numbers = []
    for pred in predictions:
        probabilities = pred.flatten()
        for idx in np.argsort(-probabilities):
            number = idx + 1
            if number not in chosen_numbers:
                chosen_numbers.append(number)
                break
    return chosen_numbers

def init_and_load_model(filepath):
    data = load_data(filepath)
    sequence_length = 10
    X, y = create_sequences(data, sequence_length=sequence_length)
    all_numbers = np.concatenate(data)
    initial_biases = adjust_initial_biases(all_numbers)
    model = lstm_model((sequence_length, 5), initial_biases)
    return model, X

# Model usage example (commented for actual deployment)
#  model, X = init_and_load_model('draw-history-full.csv')
#  single_sequence = X[0:1]
#  single_prediction = model.predict(single_sequence)
#  predicted_numbers = get_unique_numbers([single_prediction[j][0] for j in range(5)])
#  print(f"Predicted numbers: {predicted_numbers}")
