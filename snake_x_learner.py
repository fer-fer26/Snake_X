from SumTree import *
from snake_x_network import *
from snake_vs_block import *
import tensorflow.keras as K
import tensorflow as tf
import multiprocessing
import time
import random




class Memory:
    def __init__(self, arg):
        self.capacity = arg['buffer_size']
        self.tree = SumTree(self.capacity)
        self.max_p = 1
        
        #params
        self.alpha = arg['alpha']
        self.beta_initial = arg['beta_initial']
        self.beta = self.beta_initial
        self.beta_steps = arg['episode_num'] * arg['actor_num']  #experiences are sended -> update beta
        self.enable_is = arg['enable_is']
        
    def get_priority(self, td_error):
        return (td_error + 0.0001) ** self.alpha
        
    def length(self):
        return self.tree.write
        
    def add(self, experience, td_error):
        priority = self.get_priority(td_error)
        self.tree.add(priority, experience)
        
    def update(self, index, td_error):
        priority = self.get_priority(td_error)
        self.tree.update(index, priority)
        
    def beta_breaking(self):
        if self.enable_is:
            self.beta += (1 - self.beta_initial) / self.beta_steps
        
    def sample(self, batch_size):
        indexes = []
        batchs = []
        weights = np.empty(batch_size, dtype='float32')
        
        total = self.tree.total()
        section = total / batch_size
        for i in range(batch_size):
            r = section * i + random.random() * section
            (idx, priority, experience) = self.tree.get(r)

            indexes.append(idx)
            batchs.append(experience)

            if self.enable_is:
                weights[i] = (self.capacity * priority / total) ** (-self.beta)
                
            else:
                weights[i] = 1

        if self.enable_is:
            weights = weights / weights.max()

        return indexes ,batchs, weights
		
class Snake_x_learner:
    def __init__(self, arg, queue):
        self.queue = queue
        
        #network
        network = Network(arg)
        network.make_model("")
        self.model = network.model
        self.target_model = network.target_model
        
        #params
        self.actor_num = arg['actor_num']
        self.buffer_size = arg['buffer_size']
        self.initial_buffer_size = arg['initial_buffer_size']
        self.batch_size = arg['batch_size']
        self.target_update_interval = arg['target_update_interval']
        self.multi_step_count = arg['multi_step_count']
        
        self.img_shape = arg['img_shape']

        #hyper_param
        self.gamma = arg['gamma']

        self.memory = Memory(arg)
        self.end_jud = 0
        
    def get_TD_error(self, state, action, reward, next_state, weights):
        estimated = self.model.predict(state.reshape((1,) + self.img_shape + (1,)))
        future = self.target_model.predict(next_state.reshape((1,) + self.img_shape + (1,)))
        #for ddqn
        future_ = self.model.predict(next_state.reshape((1,) + self.img_shape + (1,)))
        
        return abs((reward + (self.gamma ** self.multi_step_count) * future[0][np.argmax(future_)]) * weights - estimated[0][action])

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def learning_loop(self):
        if self.memory.length() <= self.initial_buffer_size:
            #add to memory
            while not self.queue['expe'].empty():
                experiences, TD_errors = self.queue['expe'].get()
                for i in range(len(experiences)):
                    self.memory.add(experiences[i], TD_errors[i])

                #update beta of memory
                self.memory.beta_breaking()
            	
            while not self.queue['end'].empty():
                self.queue['end'].get()
                self.end_jud += 1

            if self.end_jud != self.actor_num:
                #wait
                time.sleep(1)
                return self.learning_loop()

            else:
                return self.finish()
            
        print('Training Start !!!!!!')
        total_count = 0
        while True:
            #add to memory
            while not self.queue['expe'].empty():
                experiences, TD_errors = self.queue['expe'].get()
                for i in range(len(experiences)):
                    self.memory.add(experiences[i], TD_errors[i])

                #update beta of memory
                self.memory.beta_breaking()
            
            #get data
            indexes, experiences, weights = self.memory.sample(self.batch_size)

            state = np.vstack([e.s for e in experiences]).reshape((self.batch_size,) + self.img_shape + (1,))
            next_state = np.vstack([e.n_s for e in experiences]).reshape((self.batch_size,) + self.img_shape + (1,))

            estimateds = self.model.predict(state)
            future = self.target_model.predict(next_state)
            #for ddqn
            future_ = self.model.predict(next_state)

            for i,e in enumerate(experiences):
                reward = e.r
                estimateds[i][e.a] = reward + (self.gamma ** self.multi_step_count) * future[i][np.argmax(future_[i])]
                estimateds[i][e.a] *= weights[i]

            #update model
            self.model.fit(state,estimateds,verbose=0)

            #update target model
            if total_count % self.target_update_interval == 0:
                self.update_target_model()

            #update td_error
            for i in range(len(indexes)):
                td_error = self.get_TD_error(experiences[i].s, experiences[i].a, experiences[i].r, experiences[i].n_s, weights[i])
            
                self.memory.update(indexes[i], td_error)

            #send network params
            for i in range(self.actor_num):
                if self.queue['param'][i].empty():
                    self.queue['param'][i].put((self.model.get_weights(), self.target_model.get_weights()))
            
            #count ending actor num
            while not self.queue['end'].empty():
                self.queue['end'].get()
                self.end_jud += 1
                
            #end judgement
            if self.end_jud == self.actor_num:
                break

            total_count += 1

        self.finish()

    def finish(self):
        print("save_model")   
        self.save_model()

        #send end signal
        print("Learner finish")
        self.queue['kill'][self.actor_num].put(1)

    def save_model(self):
        self.model.save_weights("param.hdf5")


        
