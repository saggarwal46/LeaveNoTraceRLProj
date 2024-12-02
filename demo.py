# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib
matplotlib.use('Agg')

import argparse
from coach_util import Agent
from env_util import get_env
from lnt import SafetyWrapper
from decays import get_qmin_func, decays
import os
from functools import partial
import logging
logging.getLogger('tensorflow').disabled = True

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import deque
import random

# Define the Q-network
class QNetworkpt(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetworkpt, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the DDQN agent
class DDQNAgentpt:
    def __init__(self, state_size, action_size, gamma=0.99, lr=0.001, batch_size=64, memory_size=10000, tau=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.tau = tau  # For soft updates
        
        # Q-network and target network
        self.qnetwork_local = QNetworkpt(state_size, action_size).cuda()
        self.qnetwork_target = QNetworkpt(state_size, action_size).cuda()
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state, epsilon=0.1):
        state = torch.FloatTensor(state).unsqueeze(0).cuda()
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        return torch.argmax(action_values).item()

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.batch_size:
            self.learn()

    def learn(self):
        # Sample a batch of experiences from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # print(f"types are {type(states[0]), type(actions), type(rewards), type(next_states), type(dones)}")
        states = torch.FloatTensor(np.array(states)).cuda()
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).cuda()
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).cuda()
        next_states = torch.FloatTensor(np.array(next_states)).cuda()
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).cuda()

        # Get Q values for the chosen actions
        q_values = self.qnetwork_local(states).gather(1, actions)
        
        # Compute the target Q values
        with torch.no_grad():
            next_actions = self.qnetwork_local(next_states).argmax(1).unsqueeze(1)
            q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions)
            q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        
        # Compute the loss
        loss = self.loss_fn(q_values, q_targets)
        
        # Perform a gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update of the target network parameters
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

def learn_safely(
    env_name,
    safety_param,
    output_dir,
    exp_name,
    schedule_type=None
    ):
    print(f"USING DECAY TYPE: {schedule_type}")
    (env, lnt_params, agent_params) = get_env(env_name, safety_param)

    log_dir = os.path.join(output_dir, exp_name)
    # 1. Create a reset agent that will reset the environment
    reset_agent = Agent(env, log_dir=os.path.join(log_dir, 'reset'), name='reset_agent', **agent_params)
    
    
    decay_params = {
        "initial_value": 0.9,
        "min_value": 1 - safety_param,
        "total_epochs": agent_params['num_training_iterations'], 
        "decay_rate": 0.99999,
    }
    q_min_func = get_qmin_func(schedule_type, safety_param, lnt_params['max_episode_steps'], **decay_params)
        

    # 2. Create a wrapper around the environment to protect it
    safe_env = SafetyWrapper(env=env,
                             log_dir=log_dir,
                             q_min_func=q_min_func,
                             reset_agent=reset_agent,
                             **lnt_params)
    # agent_params["iter_callbk"] = partial(safe_env.training_iter_callback, safe_env)
    # 3. Safely learn to solve the task.
    fw_agent = Agent(env=safe_env, log_dir=os.path.join(log_dir, 'forward'), name='forward_agent', **agent_params)
    out = fw_agent.improve()
    
    reset_agent.tf_writer_close()
    fw_agent.tf_writer_close()
    # Plot the reward and resets throughout training
    # safe_env.plot_metrics(output_dir)

def learn_aman(
    env_name,
    safety_param,
    output_dir,
    exp_name):
    print(f"im hitting this")
    (env, lnt_params, agent_params) = get_env(env_name, safety_param)

    log_dir = os.path.join(output_dir, exp_name)
    # 1. Create a reset agent that will reset the environment
    reset_agent = Agent(env, log_dir=os.path.join(log_dir, 'reset'), name='reset_agent', **agent_params)
    ch_agent = DDQNAgentpt(10,2)
    # 2. Create a wrapper around the environment to protect it
    safe_env = SafetyWrapper(env=env,
                             log_dir=log_dir,
                             q_min_func=None,
                             reset_agent=reset_agent,
                             ch_agent=ch_agent,
                             **lnt_params)

    # 3. Safely learn to solve the task.
    fw_agent = Agent(env=safe_env, log_dir=os.path.join(log_dir, 'forward'), name='forward_agent', **agent_params)
    
    print("are we even getting here")
    print(f"env is {env.reset()}")
    
    out = fw_agent.improve()
    reset_agent.tf_writer_close()
    fw_agent.tf_writer_close()
    # Plot the reward and resets throughout training
    # safe_env.plot_metrics(output_dir)
    


def learn_dangerously(
    env_name,
    safety_param,
    output_dir,
    exp_name 
    ):
    # print("LEARNING DANGEROUSY")
    log_dir = os.path.join(output_dir, exp_name)
    (env, lnt_params, agent_params) = get_env(env_name)
    safe_env = SafetyWrapper(env=env,
                             log_dir=log_dir,
                             reset_agent=None,
                             **lnt_params)
    agent = Agent(safe_env, log_dir=os.path.join(log_dir, 'forward_only'), name='agent', **agent_params)
    agent.improve()


    # 3. Safely learn to solve the task.
    out = agent.improve()
    agent.tf_writer_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Let\'s do safe RL with Leave No Trace!')
    env_list = ['cliff-walker']
    parser.add_argument('--env_name', type=str, default='cliff-walker',
                        help=('Name of the environment. The currently '
                              'supported environments are: %s') % env_list)
    parser.add_argument('--safety_param', type=float, default=0.3,
                        help=('Increasing the safety_param from 0 to 1 makes '
                              'the agent safer. A reasonable value is 0.3'))
    parser.add_argument('--output_dir', type=str, default='./tmp',
                        help='Folder for storing results')
    parser.add_argument('--learn_safely', action='store_true',
                        help=('Whether to learn safely using '
                            'Leave No Trace'))
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Experiment Name')
    parser.add_argument('--decay_type', type=str, default=None,
                        help='Safety Decay Type', choices=list(decays.keys()) + [None])
    parser.add_argument('--learn_ch', action='store_true',
                        help=('q1.5'))
    

    args = parser.parse_args()
    assert 0 < args.safety_param < 1, 'safety_param should be between 0 and 1.'
    if args.learn_safely:
        if args.learn_ch:
            learn_aman(args.env_name, args.safety_param,
                        args.output_dir, args.exp_name)
        else:
            learn_safely(args.env_name, args.safety_param,
                        args.output_dir, args.exp_name)
    else:
        learn_dangerously(args.env_name, args.safety_param,
                     args.output_dir, args.exp_name)
