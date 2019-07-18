import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from play_game import Game

model = Sequential()
model.add(Dense(128, input_shape=(66,), activation='relu'))
model.add(Dense(96, activation='relu'))
model.add(Dense(22, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])



for x in range(10):
    monopoly = Game(model, 1 - x * 0.1)
    monopoly.play_games(25)
    X, Y = monopoly.get_observations()
    model.fit(x=X, y=Y, epochs=10, validation_split=0.2, verbose=0)
    print(x)

model.fit(x=X, y=Y, epochs=10, validation_split=0.2, verbose=1)

print("trained!")

monopoly = Game(model, 0)
monopoly.play_games(1)