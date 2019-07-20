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
    model.add(Dense(22, input_shape=(66,), activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    #model.load_weights('model.h5')
    return model

def get_end_balances(model):
    test_game = Game(model, 0, False)
    test_game.set_up_game([model, model, model])
    test_game.players[1].epsilon = 1
    test_game.players[2].epsilon = 1
    test_game.simulate_game()
    print("Model: " + str(test_game.players[0].funds) + ", Random 1: " + str(test_game.players[1].funds) + ", Random 2: " + str(test_game.players[2].funds))

model = monopoly_model()
get_end_balances(model)
training_runs = 1
while(True):
    for x in range(4):
        print("ε=0")
        monopoly = Game(model, 0, False)
        monopoly.play_games(1)
        X, Y = monopoly.get_observations()
        for example in range(len(X)):
            newX = np.array([X[example]])
            newY = np.array([Y[example]])
            model.fit(x=newX, y=newY, epochs=10, verbose=0, callbacks=[tensorboard_callback])
        #model.fit(x=X, y=Y, epochs=10, verbose=1, callbacks=[tensorboard_callback])
        model.save_weights('model.h5')
        get_end_balances(model)
    print("Finished training run " + str(training_runs))
    training_runs += 1

print("Training complete")