import numpy as np
import random
from collections import deque, namedtuple
from snake_x_network import *
from snake_x_logger import *
import tensorflow as tf
import tensorflow.keras as K
from PIL import Image
from copy import deepcopy
import gym
import copy
from matplotlib import pyplot



Experience = namedtuple("Experience",["s","a","r","n_s"])


class Snake_x_actor:
    def __init__(self, arg, queue):
        self.queue = queue
        
        #params
        self.buffer_size = arg['buffer_size']
        self.batch_size = arg['batch_size']
        self.episode_num = arg['episode_num']
        self.multi_step_count = arg['multi_step_count']
        self.actor_num = arg['actor_num']
        self.actor_count = arg['actor_count']

        #hyper_param
        self.gamma = arg['gamma']
        self.epsilon = arg['epsilon']

        self.gamma_list = [self.gamma ** i for i in range(self.multi_step_count)]

        self.game = arg['game']
        self.img_shape = arg['img_shape']
        self.enable_img = arg['enable_img']
        self.action_num = arg['action_num']
        
        #make model
        network = Network(arg)
        network.make_model("")
        self.model = network.model
        self.target_model = network.target_model

        self.update_count = 0
        
        #logger
        self.logger = Snake_x_logger(arg)

    def grayscale(self, state):
        if self.enable_img:
            state = Image.fromarray(state).convert("L")
            state = np.array(state)

            #normalize
            state = state / 255.0

        return state

    def predict(self, state):
        x = self.model.predict(state.reshape((1,) + self.img_shape + (1,)))

        return np.argmax(x)

    def policy(self, state):
        if np.random.random() < self.epsilon:
            return random.choice(range(self.action_num))

        else:
            estimated = self.predict(state)

            return estimated

    def get_TD_error(self, state, action, reward, next_state):
        estimated = self.model.predict(state.reshape((1,) + self.img_shape + (1,)))
        future = self.target_model.predict(next_state.reshape((1,) + self.img_shape + (1,)))
        #for ddqn
        future_ = self.model.predict(next_state.reshape((1,) + self.img_shape + (1,)))
        
        return abs(reward + (self.gamma ** self.multi_step_count) * future[0][np.argmax(future_)] - estimated[0][action])

    def train_loop(self):
        for ep in range(self.episode_num):
            total_reward = 0

            self.state_buffer = deque(maxlen=self.multi_step_count)
            self.reward_buffer = deque(maxlen=self.multi_step_count)
            self.action_buffer = deque(maxlen=self.multi_step_count)
            
            #for sending to learner
            experiences = []
            TD_errors = []
            
            state = self.game.reset()
            state = self.grayscale(state)
            while True:
                action = self.policy(state)
                next_state, reward, done, _ = self.game.step(action)
                
                self.reward_buffer.append(reward)
                self.state_buffer.append(state)
                self.action_buffer.append(action)

                if done:
                    break

                total_reward += reward

                next_state = self.grayscale(next_state)

                #store experiences
                if len(self.reward_buffer) == self.multi_step_count:
                    reward = sum([self.reward_buffer[i] * self.gamma_list[i] for i in range(self.multi_step_count)])
                    e = Experience(self.state_buffer[0], self.action_buffer[0], reward, next_state)
                    experiences.append(e)
                    
                    td_error = self.get_TD_error(self.state_buffer[0], self.action_buffer[0], reward, next_state)
                    TD_errors.append(td_error)

                if done:
                    break

                state = next_state

            #send date to learner
            self.queue['expe'].put((experiences, TD_errors))
            
            #update network param
            if not self.queue['param'][self.actor_count].empty():
                model_weights, target_model_weights = self.queue['param'][self.actor_count].get()

                #set
                self.model.set_weights(model_weights)
                self.target_model.set_weights(target_model_weights)

            #log
            if self.actor_count == self.actor_num - 1:
                self.logger.store_reward(total_reward, self.game.score, ep)

        #finish
        self.queue['end'].put(1)

        #show graph
        if self.actor_count == self.actor_num - 1:
            self.logger.show_graph()

        #send end signal
        print("Actor " + str(self.actor_count) + " finish")
        self.queue['kill'][self.actor_count].put(1)
