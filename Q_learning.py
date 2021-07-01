import tensorflow as tf
import numpy as np
import random
import gym
from importlib import reload  



from gym.envs.registration import register 
tf.reset_default_graph()
reload(gym.envs.registration)

# =============================================================================
# class Environment():
#     def __init__(self):
#         pass
#         # construct the environment where agent can perceive and act.
#     def FrozenLakeNoSlippery(self):
#         pass
#         # construct frozen lake without slippery
# 
# class DeepQAgent():
# 
#     def __init__(self, args, env):
#         pass
#         # setting hyper-parameters and initialize NN model
#         
#     def _nn_model(self, env):
#         pass
#         # build nn model
#         
#     def train(self):
#         pass
#         # training the agent
# 
#     def test(self, Q):
#         pass
#         # testing the agent
#         
#     def displayQ():
#         pass
# =============================================================================

class Environment():
    def __init__(self):
        pass
    def FrozenLakeNoSlippery(self):
        register(
                 id= 'FrozenLakeNoSlippery-v0',
                 entry_point='gym.envs.toy_text:FrozenLakeEnv',
                 kwargs={'map_name' : '4x4', 'is_slippery': False},
                 max_episode_steps=100,
                 reward_threshold=0.82
                 )
        env = gym.make('FrozenLakeNoSlippery-v0')
        return env


class DeepQAgent():
    def __init__(self,env):
        # set hyperparameters
        self.max_episodes = 10
        self.max_actions = 7
        self.discount = 0.4
        self.exploration_rate = 1.0
        self.exploration_decay = 1.0/10
        # get envirionment
        self.env = env
    
        # nn_model parameters
        self.in_units = env.observation_space.n
        self.out_units = env.action_space.n
        self.hidden_units = 4
        
        # construct nn model
        self._nn_model()
    
        # save nn model
        self.saver = tf.train.Saver()    
    def _nn_model(self):
        self.a0 = tf.placeholder(tf.float32, shape=[1, self.in_units]) # input layer
        self.y = tf.placeholder(tf.float32, shape=[1, self.out_units]) # ouput layer
        
        # from input layer to hidden layer
        self.w1 = tf.Variable(tf.zeros([self.in_units, self.hidden_units], dtype=tf.float32)) # weight
        self.b1 = tf.Variable(tf.random_uniform([self.hidden_units], 0, 0.01, dtype=tf.float32)) # bias
        self.a1 = tf.nn.relu(tf.matmul(self.a0, self.w1) + self.b1) # the ouput of hidden layer
        
        # from hidden layer to output layer
        self.w2 = tf.Variable(tf.zeros([self.hidden_units, self.out_units], dtype=tf.float32)) # weight
        self.b2 = tf.Variable(tf.random_uniform([self.out_units], 0, 0.01, dtype=tf.float32)) # bias
        
        # Q-value and Action
        self.a2 = tf.matmul(self.a1, self.w2) + self.b2 # the predicted_y (Q-value) of four actions
        self.action = tf.argmax(self.a2, 1) # the agent would take the action which has maximum Q-value
    
        # loss function
        self.loss = tf.reduce_sum(tf.square(self.a2-self.y))
        
        # upate model
        self.update_model =  tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(self.loss)
        
    def train(self):
    # get hyper parameters
        max_episodes = self.max_episodes
        max_actions = self.max_actions
        discount = self.discount
        exploration_rate = self.exploration_rate
        exploration_decay = self.exploration_decay
        
        # start training
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer()) # initialize tf variables
            for i in range(max_episodes):
                
                state = self.env.reset() # reset the environment per eisodes
                for j in range(max_actions):
                     # get action and Q-values of all actions
                    action, pred_Q = sess.run([self.action, self.a2],feed_dict={self.a0:np.eye(16)[state:state+1]})
                    
                    # if explorating, then taking a random action instead
                    if np.random.rand()<exploration_rate: 
                        action[0] = self.env.action_space.sample() 
                        action[0] = random.randint(0,2)

                    # get nextQ in given next_state
                    next_state, rewards, done, info = self.env.step(action[0])
                    next_Q = sess.run(self.a2,feed_dict={self.a0:np.eye(16)[next_state:next_state+1]})

                    # update
                    update_Q = pred_Q
                    update_Q [0,action[0]] = rewards + discount*np.max(next_Q)
                    
                    sess.run([self.update_model],
                             feed_dict={self.a0:np.identity(16)[state:state+1],self.y:update_Q})
                    state = next_state
                    print('action{},pred_Q{},state{},next_state{},rewards{},info{},next_Q{}'.format(action,pred_Q,state,next_state,rewards,info,next_Q))
                    #print('next_state={0},rewards={1},update_Q={2}'.format(next_state, rewards, update_Q))
                     # if fall in the hole or arrive to the goal, then this episode is terminated.
                    if done:
                        print('done, rewark = {}, exploration_rate={}'.format(rewards,exploration_rate))
                        if exploration_rate > 0.001:
                            exploration_rate -= exploration_decay
                        break
            # save model
            save_path = self.saver.save(sess, "./nn_model.ckpt")    
    def test(self):
        # get hyper-parameters
        max_actions = self.max_actions
        # start testing
        with tf.Session() as sess:
            # restore the model
            sess.run(tf.global_variables_initializer())
            saver=tf.train.import_meta_graph("./nn_model.ckpt.meta") # restore model
            saver.restore(sess, tf.train.latest_checkpoint('./'))# restore variables
            
            # testing result
            state = self.env.reset()
            for j in range(max_actions):
                self.env.render() # show the environments
                # always take optimal action
                action, pred_Q = sess.run([self.action, self.a2],feed_dict={self.a0:np.eye(16)[state:state+1]})
                if np.random.rand()<0.2: 
                    pass
                    #action[0] = self.env.action_space.sample() 
                # update
                next_state, rewards, done, info = self.env.step(action[0])
                state = next_state

                if done:
                    self.env.render()
                    break




if __name__ == '__main__':
    env = Environment().FrozenLakeNoSlippery() # construct the environment
    agent = DeepQAgent(env) # get agent
    print("START TRAINING...")
    agent.train()
    print("\n\nTEST\n\n")
    #agent.test()



