# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 20:25:54 2020

@author: DANH
"""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random
from IPython.display import clear_output
from collections import deque
import progressbar

import gym

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape, Flatten, Input
from tensorflow.keras.optimizers import Adam
#reload(gym)
import time

class Environment:

    ACTION_SPACE = [-1,1,-1] #0 = keep, 1 = buy, -1 = sale

    def __init__(self):

        self.current_step = 0
        self.rewards = 0
    
    def set_env(self,states=[],rewards=[]):
        self.states = states
        self.rewards = rewards
        self.num_state = len(self.states)
        
    def reset(self):
        self.current_step = 0
        return self.states[0]
    
    def step(self, action):

        if (self.current_step == self.num_state-1):
            return 0, 0, True #terminated
        #print(self.current_step)
        next_state = self.states[self.current_step+1]
        reward = self.rewards[self.current_step] * self.ACTION_SPACE[action] 
                    
        self.current_step +=1
        return next_state, reward, False #terminated
           

class Agent:
    def __init__(self, enviroment, optimizer):
        
        # Initialize atributes
        self._state_size = enviroment.num_state
        self._action_size = len(enviroment.ACTION_SPACE)
        self._optimizer = optimizer
        self.n_input = 20
        self.n_features = 1
        self.expirience_replay = deque(maxlen=70)
        
        # Initialize discount and exploration rate
        self.gamma = 0.5
        self.epsilon = 0.2
        
        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.alighn_target_model()

    def store(self, state, action, reward, next_state, terminated):
        self.expirience_replay.append((state, action, reward, next_state, terminated))
    
    def _build_compile_model(self):
        #model = Sequential()
        #model.add(Embedding(self._state_size, 10, input_length=1))
        #model.add(Reshape((10,)))
        #model.add(Dense(20, input_shape=(self.n_input, ), activation='relu'))
        #model.add(Flatten())
        #model.add(Dense(50, activation='relu'))
        #model.add(Dense(3, activation='softmax'))        
        #model.compile(loss='mse', optimizer=self._optimizer)
        inputs = Input(shape=(20,))
        x = Dense(10,activation='relu')(inputs)
        outputs = Dense(3,activation='softmax')(x) 
        model = Model(inputs,outputs)
        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return np.random.randint(0,self._action_size)
        
        q_values = self.q_network.predict(np.asarray([state]))
        return np.argmax(q_values[0])
    
    def getact(self, state):
        q_values = self.target_network.predict(np.asarray([state]))
        return np.argmax(q_values[0])

    def retrain(self, batch_size):
        minibatch = random.sample(self.expirience_replay, batch_size)
        
        for state, action, reward, next_state, terminated in minibatch:
            
            target = self.q_network.predict(np.asarray([state]))
            
            if terminated:
                target[0][action] = reward
            else:
                t = self.target_network.predict(np.asarray([next_state]))
                target[0][action] = reward + self.gamma * np.amax(t)
            
            self.q_network.fit(np.asarray([state]), target, epochs=1, verbose=0)
      
 
def makeTimeSerialData(data=[],timestep=20,num_predict=10):
    X_train=[]   
    Y_train = []
    batchsize = data.shape[0]
    for i in range(batchsize-timestep-num_predict):
        state = data[i:i+timestep,:]
        reward = (np.max(data[i+timestep + 3: i+timestep + 3 + num_predict ,0:1]) - state[-1])/state[-1]
        X_train.append(state)    
        Y_train.append(reward)
    return np.asarray(X_train).reshape(-1,timestep), np.asarray(Y_train)
          

Stock = pd.read_csv(r'E:\StocksData\MWG.csv',names=['date','Open','High','Low','Close','Volume','g'])

Stock = Stock.drop('g',axis=1)
StockClose = Stock['Close'].values.reshape(-1,1)
StockClose = StockClose[-500:,:].reshape(-1,1)
StockClose = StockClose/np.max(StockClose)

X_train,Y_train = makeTimeSerialData(data=StockClose)

#np.max(Y_train)
#np.argmax(Y_train)

enviroment = Environment()
enviroment.set_env(states=X_train[-470:-430],rewards=Y_train[-470:-430])
optimizer = Adam(learning_rate=0.001)
agent = Agent(enviroment, optimizer)

batch_size = 20
num_of_episodes = 500
timesteps_per_episode = enviroment.num_state
agent.q_network.summary()    

#s = np.asarray([enviroment.reset()])


#action = agent.act(X[131])
#np.shape(X_train[3])
#X = X_train.reshape(-1,20)
#a = X_train[1]
#q = agent.q_network.predict(s)

#np.shape(X_train[3])
total_reward = []
for e in range(0, num_of_episodes):
    total_reward = []
    # Reset the enviroment
    state = enviroment.reset()
    #state = np.reshape(state, [1, 20])
    
    # Initialize variables
    reward = 0
    terminated = False
    
    #bar = progressbar.ProgressBar(maxval=timesteps_per_episode/10, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    #bar.start()
    print(e)
    for timestep in range(timesteps_per_episode):
        # Run Action
        action = agent.act(state)
        
        # Take action    
        next_state, reward, terminated = enviroment.step(action) 
        
        #next_state = np.reshape(next_state, [1, 1])
        agent.store(state, action, reward, next_state, terminated)
        total_reward.append((action,float(reward)))
        state = next_state
        
        if terminated:
            agent.alighn_target_model()
            break
            
        if (len(agent.expirience_replay) % batch_size == 0):
            agent.retrain(batch_size)
        
        if timestep%100 == 0:
            pass
            #bar.update(timestep/100 + 1)
    
    #bar.finish()
    #if (e + 1) % 10 == 0:
       # print("**********************************")
        #print("Episode: {}".format(e + 1))
        #enviroment.render()
        #print("**********************************")


#a = np.asarray(X_train)
r=0
for i in total_reward:
    if i[0]==1 and i[1]!=0:
        r +=i[1]
        
#===========

enviroment.set_env(states=X_train[420:460],rewards=Y_train[420:460])
#optimizer = Adam(learning_rate=0.01)
total_reward = []
for e in range(0, 1):
    # Reset the enviroment
    state = enviroment.reset()
    #state = np.reshape(state, [1, 20])
    
    # Initialize variables
    reward = 0
    terminated = False
    
    bar = progressbar.ProgressBar(maxval=timesteps_per_episode, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    #bar.start()
    
    for timestep in range(timesteps_per_episode):
        # Run Action
        action = agent.getact(state)
        
        # Take action    
        next_state, reward, terminated = enviroment.step(action) 
        #next_state = np.reshape(next_state, [1, 1])
        
        total_reward.append((action,float(reward)))
        state = next_state
        
        
        if timestep%1 == 0:
            pass
            #bar.update(timestep/10 + 1)
    
    #bar.finish()
    #if (e + 1) % 10 == 0:
       # print("**********************************")
        #print("Episode: {}".format(e + 1))
        #enviroment.render()
        #print("**********************************")


    
plt.plot(StockClose[-460:-420])  
plt.plot(Y_train[-390:-350])  
np.sum(Y_train,axis=0)
    

