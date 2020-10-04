import os
import multiprocessing
from snake_x_actor import *
from snake_x_learner import *
from snake_vs_block import *
from snake_x_logger import *
import time
from copy import deepcopy




def start_actor(arg, queue):
    #add game arg
    arg['game'] = Snake_vs_block(arg)

    enable_gpu = arg['enable_gpu']
    if enable_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    actor = Snake_x_actor(arg, queue)
    actor.train_loop()
	
def start_learner(arg, queue):
    learner = Snake_x_learner(arg, queue)
    learner.learning_loop()


	
if __name__ == '__main__':
    arg = {
      #params
      'actor_num' : 8,
      'actor_count' : 0,
      'buffer_size' : 50000,
      'initial_buffer_size' : 10000,
      'batch_size' : 2000,
      'episode_num' : 5000,
      'target_update_interval' : 50,
      'multi_step_count' : 1,
      'enable_gpu' : True,
      
      #hyper_params
      'gamma' : 0.99,
      'learning_late' : 1e-3,
      'epsilon' : 0,

      #epsilon
      'ep_epsilon' : 0.4,
      'ep_alpha' : 5,
      
      #memory_parama
      'alpha' : 1.0,
      'beta_initial' : 0.0,
      'enable_is' : False,
      
      #log
      'log_interval' : 10,
      'print_log' : True,
      'show_graph' : True,
      
      #others
      'enable_img' : False,
      'play' : False,  #show playing
      'draw_screen' : False,
    }
    
    #add params of env
    enable_img = arg['enable_img']
    
    if enable_img:
        arg['img_shape'] = (113, 200)
        arg['action_num'] = 5
        
    else:
        arg['img_shape'] = (5, 24)
        arg['action_num'] = 5


    actor_num = arg['actor_num']
            
    #queue (param, experience, end)
    param = [multiprocessing.Queue() for i in range(actor_num)]
    kill_queue = [multiprocessing.Queue() for i in range(actor_num + 1)]
    queue = {
      'param' : param,
      'expe' : multiprocessing.Queue(),
      'end' : multiprocessing.Queue(),
      'kill' : kill_queue,
    }

    #produce epsilon
    ep_epsilon = arg['ep_epsilon']
    ep_alpha = arg['ep_alpha']
    epsilon = []
    for i in range(actor_num):
        if actor_num == 1:
            ep = 0.1

        else:
            ep = ep_epsilon ** (1 + i / (actor_num - 1) * ep_alpha)
        epsilon.append(ep)
       
    processes = []
    
    #actor
    args = [0 for i in range(actor_num)]
    for actor in range(actor_num):
        args[actor] = deepcopy(arg)
        args[actor]['actor_count'] = actor
        args[actor]['epsilon'] = epsilon[actor]
        actor_process = multiprocessing.Process(target=start_actor, args=(args[actor], queue))
        processes.append(actor_process)

    #learner
    learner_process = multiprocessing.Process(target=start_learner, args=(arg, queue))
    processes.append(learner_process)
    
    #start
    for process in processes:
        process.start()

    #kill processes
    kill_count = 0
    while kill_count < actor_num + 1:
        for i in range(actor_num + 1):
            if not queue['kill'][i].empty():
                queue['kill'][i].get()
                processes[i].terminate()
                kill_count += 1
        time.sleep(1)
    
    