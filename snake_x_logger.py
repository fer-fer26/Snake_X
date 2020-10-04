from matplotlib import pyplot




class Snake_x_logger:
    def __init__(self, arg):
        self.log_interval = arg['log_interval']
        self.print_log = arg['print_log']
        self.show = arg['show_graph']
        
        self.reward_buffer = []
        
        self.total_reward = 0
        self.total_score = 0
        self.episode = 0
        
        self.max_score = 0
        
    def store_reward(self, reward, score, episode):
        self.total_reward += reward
        self.total_score += score
        
        self.episode = episode
        
        if score > self.max_score:
           self.max_score = score
        
        if episode % self.log_interval == 0:
            self.reward_buffer.append(self.total_reward / self.log_interval)
            
            if self.print_log:
                print("====================================")
                print("episode : " + str(self.episode))
                print("reward : " + str(self.total_reward / self.log_interval))
                print("score : " + str(self.total_score / self.log_interval))
                print("max_score : " + str(self.max_score))
                print("====================================")
                
            self.total_reward = 0
            self.total_score = 0
            self.max_score = 0
        
    def show_graph(self):
        x = [i for i in range(len(self.reward_buffer))]
        pyplot.plot(x, self.reward_buffer)
        
        if self.show:
            pyplot.show()
