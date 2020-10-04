import tensorflow.keras as K
import numpy as np
from snake_vs_block import *
from snake_x_network import *
import sys
import h5py




class Snake_x_play:
    def __init__(self, arg):
        arg['play'] = True
        arg['draw_screen'] = True
        arg['learning_late'] = 1e-3

        self.game = Snake_vs_block(arg)
        self.play_num = arg['play_num']

        enable_img = arg['enable_img']
        if enable_img:
            arg['img_shape'] = (113, 200)
            arg['action_num'] = 11

        else:
            arg['img_shape'] = (5, 24)
            arg['action_num'] = 5

        self.img_shape = arg['img_shape']

        #load model
        network = Network(arg)
        network.make_model("param.hdf5")
        self.model = network.model

    def predict(self, state):
        state = state.reshape((1,) + self.img_shape + (1,))
        estimated = self.model.predict(state)

        return np.argmax(estimated)

    def play(self):
        for ep in range(self.play_num):
            state = self.game.reset()
            while True:
                action = self.predict(state)
                next_state, reward, done, _ = self.game.step(action)

                if done:
                    break

                state = next_state

            print("episode : " + str(ep) + " --- Score is " + str(self.game.score))



if __name__ == "__main__":
    arg = {
        'play_num' : 100,
        'enable_img' : False,
    }

    snake_x_play = Snake_x_play(arg)
    snake_x_play.play()
            
