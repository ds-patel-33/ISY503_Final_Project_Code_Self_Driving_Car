import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import glorot_uniform, he_uniform
from keras.src.utils.vis_utils import plot_model
tf.random.set_seed(27)
X_train = np.load("X_train.npy")
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")
print("Shape of the images: " + str(X_train[0].shape))
print("Example of measurement: " + str(y_train[0]))
if not os.path.exists("models"): os.makedirs("models")
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(76, 320, 3),
    
)

inputs = keras.Input(shape=(76, 320, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.25)(x)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
base_model.trainable = False
model.summary()
epochs = 50
batch_size = 32
learning_rate = 0.0001

model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              loss=keras.losses.mse,
              metrics=['mse'])

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mse',
                                                 factor=0.5,
                                                 patience=5,
                                                 min_lr=0.1*learning_rate,
                                                 verbose=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=2, verbose=1)

history = model.fit(X_train,
                    y_train,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    callbacks=[reduce_lr, early_stopping_callback],
                    shuffle=True,
                    epochs=epochs,
                    verbose=1)
end_epoch_top_layer_training = len(history.history['loss'])
base_model.trainable = True
model.summary()
epochs = 20
batch_size = 32
learning_rate /= 10
filepath = "models/model_ft_{epoch:02d}_{val_mse:.4f}.h5"

model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              loss=keras.losses.mse,
              metrics=['mse'])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                            filepath=filepath,
                                            monitor='val_mse',
                                            mode='min',
                                            verbose=1,
                                            save_best_only=True)
history_ft = model.fit(X_train,
                    y_train,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    callbacks=[reduce_lr, early_stopping_callback, model_checkpoint_callback],
                    shuffle=True,
                    epochs=epochs,
                    verbose=1)
fig, axs = plt.subplots(2, 1, figsize=(15, 15))

history.history['loss'].extend(history_ft.history['loss'])
history.history['val_loss'].extend(history_ft.history['val_loss'])
history.history['mse'].extend(history_ft.history['mse'])
history.history['val_mse'].extend(history_ft.history['val_mse'])
epochs = np.arange(1, len(history.history['loss']) + 1)

axs[0].plot(epochs, history.history['loss'])
axs[0].plot(epochs, history.history['val_loss'])
axs[0].axvline(end_epoch_top_layer_training, color='r')
axs[0].title.set_text('Loss during training')
axs[0].legend(['Training set', 'Validation set', "Start of fine tuning"])

axs[1].plot(epochs, history.history['mse'])
axs[1].plot(epochs, history.history['val_mse'])
axs[1].axvline(end_epoch_top_layer_training, color='r')
axs[1].title.set_text('MSE during training')
axs[1].legend(['Training set', 'Validation set', "Start of fine tuning"])

plt.show()
