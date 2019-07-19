import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from play_game import Game
from datetime import datetime
import keras

logdir="logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

def monopoly_model():
    model = Sequential()
    model.add(Dense(128, input_shape=(66,), activation='relu'))
    model.add(Dense(22, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    #model.load_weights('model.h5')
    return model

def get_end_balances(model, prev_model_1, prev_model_2):
    test_game = Game(model, 1, False)
    test_game.set_up_game([model, prev_model_1, prev_model_2])
    test_game.simulate_game()
    print("Model: " + str(test_game.players[0].funds) + ", Random 1: " + str(test_game.players[1].funds) + ", Random 2: " + str(test_game.players[2].funds))

model_history = [None, None, None]
model = monopoly_model()
training_runs = 1
while(True):
    for x in range(100):
        monopoly = Game(model, 0.4, False)
        monopoly.play_games(32)
        X, Y = monopoly.get_observations()
        model.fit(x=X, y=Y, epochs=10, validation_split=0.1, verbose=1, callbacks=[tensorboard_callback])
        model.save_weights('model.h5')
        model_history[0] = model_history[1]
        model_history[1] = model_history[2]
        model_history[2] = model
        if x > 3:
            get_end_balances(model_history[2], model_history[1], model_history[0])
            get_end_balances(model_history[2], model_history[1], model_history[0])
    print("Finished training run " + str(training_runs))
    training_runs += 1

print("Training complete")