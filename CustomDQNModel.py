import tensorflow as tf

class CustomDQNModel(tf.keras.Model):
    def __init__(self, n_timesteps, n_features, n_outputs):
        super(CustomDQNModel, self).__init__()
        self.lstm1 = tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(n_timesteps, n_features))
        self.lstm2 = tf.keras.layers.LSTM(50)
        self.dense1 = tf.keras.layers.Dense(50, activation='relu')
        self.dense2 = tf.keras.layers.Dense(n_outputs, activation='softmax')

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = self.dense1(x)
        output = self.dense2(x)
        return output
