import tensorflow as tf
from tensorflow.keras import layers

def build_model(input_shape, output_dim):
    model = tf.keras.models.Sequential()
    model.add(layers.Conv2D(32, (4,8), activation=tf.nn.relu, input_shape=input_shape, padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(32, (2,4), activation=tf.nn.relu, padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(32, (2,4), activation=tf.nn.relu, padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation=tf.nn.relu))
    model.add(layers.Dense(output_dim, activation='softmax'))
    return model

def init_model(class_num=5, lat=24, lon=72, var_num=4, lr=0.0001):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = tf.keras.metrics.CategoricalAccuracy()
    model = build_model((lat, lon, var_num), class_num)
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    return model

if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    class_num = 5
    lat, lon = 24, 72
    var_num = 4
    model = build_model((lat, lon, var_num), class_num)
    model.summary()

