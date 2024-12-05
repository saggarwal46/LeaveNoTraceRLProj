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


from coach_util import Transition, RunPhase
from gym import Wrapper
from gym.wrappers.time_limit import TimeLimit
import numpy as np
import json
import os

import torch

import tensorflow as tf

import ipdb

class SafetyWrapper(Wrapper):
    # TODO: allow user to specify number of reset attempts. Currently fixed at 1.
    def __init__(self, 
                 env,
                 log_dir,
                 q_min_func,
                 reset_agent,
                 reset_reward_fn,
                 reset_done_fn,
                 ch_agent,
                 **kwargs,
                 ):
        '''
        A SafetyWrapper protects the inner environment from danerous actions.

        args:
            env: Environment implementing the Gym API
            reset_agent: an agent implementing the coach Agent API
            reset_reward_fn: a function that returns the reset agent reward
                for a given observation.
            reset_done_fn: a function that returns whether the reset agent
                has successfully reset.
            q_min: a float that is used to decide when to do early aborts.
        '''
        assert isinstance(env, TimeLimit)
        super(SafetyWrapper, self).__init__(env)
        self._reset_agent = reset_agent
        if reset_agent is not None:
            self._reset_agent.exploration_policy.change_phase(RunPhase.TRAIN)
        # print("RESET AGENT", reset_agent)
        self.env._reset_reward_fn = reset_reward_fn
        self.env._reset_done_fn = reset_done_fn
        self._max_episode_steps = env._max_episode_steps
        self.q_min_func = q_min_func
        # self._q_min = q_min
        self._obs = env.reset()
        # print(f"observation is {self._obs}")/
        self.ch_agent = ch_agent
        # Setup internal structures for logging metrics.
        self._total_resets = 0  # Total resets taken during training
        self._episode_rewards = []  # Rewards for the current episode
        self._reset_history = []
        self._reward_history = []
        self._training_iter = 0

        self._pusher_reset_cnt = 0
        self._pusher_forward_cnt = 0
        
        self._episode_count = 0
        
        self._falling_off_cliff_reset_cnt = 0
        self._falling_on_cliff_reset_cnt = 0
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir
        # self.session = tf.Session(config=tf.ConfigProto(
        #     log_device_placement=True, allow_soft_placement=True))
        # self.summary_writer = tf.summary.FileWriter(log_dir)
        # self.summary_writer = tf.summary.FileWriter(log_dir)
        # self.sess = tf.Session()

        self.epsilon = 1
        self.epsilon_min=0.01
        self.epsilon_decay=0.995

    def _reset(self):
        '''Internal implementation of reset() that returns additional info.'''
        # print("RESETING BRO")
        obs = self._obs
        obs_vec = [np.argmax(obs)]
        for t in range(self._max_episode_steps):
            (reset_action, _) = self._reset_agent.choose_action(
                {'observation': obs[:, None]}, phase=RunPhase.TRAIN)
            # print("TAKING A STEP WOOO")
            (next_obs, r, _, info) = self.env.step(reset_action)
            reset_reward = self.env._reset_reward_fn(next_obs, reset_action)
            reset_done = self.env._reset_done_fn(next_obs)
            transition = Transition({'observation': obs[:, None]},
                                    reset_action, reset_reward,
                                    {'observation': next_obs[:, None]},
                                    reset_done)
            self._reset_agent.memory.store(transition)
            obs = next_obs
            obs_vec.append(np.argmax(obs))
            memory_size = self._reset_agent.memory.num_transitions_in_complete_episodes()
            if memory_size > self._reset_agent.tp.batch_size:
                # Do one training iteration of the reset agent
                self._reset_agent.train()
            if reset_done:
                break
        if not reset_done:
            curr_obs = obs
            obs = self.env.reset()
            self._total_resets += 1
            # if self.env.env.falling_off_cliff(curr_obs):
            #     self._falling_off_cliff_reset_cnt += 1
            # if self.env.env.fell_on_cliff(curr_obs):
            #     self._falling_on_cliff_reset_cnt += 1
            
            
        # Log metrics
        self._reset_history.append(self._total_resets)
        self._reward_history.append(np.mean(self._episode_rewards))
        self._episode_count += 1

        self._episode_rewards = []

        # If the agent takes an action that causes an early abort the agent
        # shouldn't believe that the episode terminates. Because the reward is
        # negative, the agent would be incentivized to do early aborts as
        # quickly as possible. Thus, we set done = False.
        done = False

        # Reset the elapsed steps back to 0
        self.env._elapsed_steps = 0
        return (obs, r, done, info)

    def reset(self):
        if self._reset_agent is None:
            # print("RESET HERE")
            obs = self.env.reset()
            self._total_resets += 1
            if self.env.env.falling_off_cliff(self._obs):
                self._falling_off_cliff_reset_cnt += 1
            if self.env.env.fell_on_cliff(self._obs):
                self._falling_on_cliff_reset_cnt += 1
            # print(self._total_resets)
            return obs
        (obs, r, done, info) = self._reset()
        return obs

    def step(self, action):
        if self._reset_agent is not None: 
            reset_q = self._reset_agent.get_q(self._obs, action)
            # forward_q = self._forward_agent.get_q(self._obs,action)
            # state = [env ,forward_q , reset_q]
            # epsilon = 1
            # epsilon_min=0.01
            # epsilon_decay=0.995
            

            if self._training_iter < 700000: # original code
                # print(f"Using original scheduler")
                # next_state, reward, done, _ = env.step(action)
                # print(f"USING TRAINING ITER: {self._training_iter}")
                q_min_thresh, _ = self.q_min_func(self._training_iter)
                q_min = -1 * (1. - 0.3) * self._max_episode_steps
                if reset_q < q_min_thresh: # if action == 0
                    print(f"IS THIS HITTING")
                    (obs, r, done, info) = self._reset()
                    self._obs = obs
                else:
                    (obs, r, done, info) = self.env.step(action)
                    self._episode_rewards.append(r)
                self._obs = obs
                return (obs, r, done, info)
            else: # choose agent
                # print(f"Using choose policy")
                initial_state = self._obs
                action2 = self.ch_agent.act(self._obs, self.epsilon) # 0 or 1
                # print(f"Action is {action2} and type {type(action2)}")
                if action2 == 0: # if action == 0
                    # print(f"selected reset policy")
                    (obs, r, done, info) = self._reset()
                    self._obs = obs
                    ch_reward = r
                    self._pusher_reset_cnt += 1
                    # if r==0:
                    #     ch_reward = 1.0
                # else:
                    #     ch_reward = 0.0
                else:
                    # print(f"selected forward policyx")
                    (obs, r, done, info) = self.env.step(action)
                    self._episode_rewards.append(r)
                    self._obs = obs
                    ch_reward = r
                    self._pusher_forward_cnt += 1

                self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
                # reward agent for being good boy
                # ch_reward = torch.Tensor([5])
                # ch_reward = r
                # ipdb.set_trace()
                # print(f"about to be good boy")
                self.ch_agent.step(state=initial_state, action=action2, reward=ch_reward, next_state=self._obs, done=done)
                # print(f"became good boy")
                # done = True # it works
                return (obs, r, done, info)
        
        

            # # next_state, reward, done, _ = env.step(action)
            # # print(f"USING TRAINING ITER: {self._training_iter}")
            # # q_min_thresh, _ = self.q_min_func(self._training_iter)
            # q_min = -1 * (1. - 0.3) * self._max_episode_steps
            # if reset_q < q_min: # if action == 0
            #     (obs, r, done, info) = self._reset()
            #     self._obs = obs
            # else:
            #     (obs, r, done, info) = self.env.step(action)
            #     self._episode_rewards.append(r)
            # self._obs = obs
            # return (obs, r, done, info)
        # print("STEP HERE")
        (obs, r, done, info) = self.env.step(action)
        self._episode_rewards.append(r)
        self._obs = obs
        return (obs, r, done, info)

    def plot_metrics(self, output_dir='/tmp'):
        '''
        Plot metrics collected during training.

        args:
            output_dir: (optional) folder path for saving results.
        '''

        import matplotlib.pyplot as plt


        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        data = {
            'reward_history': self._reward_history,
            'reset_history': self._reset_history
        }
        with open(os.path.join(self.log_dir, 'data.json'), 'w') as f:
            json.dump(data, f)

        # Prepare data for plotting
        rewards = np.array(self._reward_history)
        lnt_resets = np.array(self._reset_history)
        num_episodes = len(rewards)
        baseline_resets = np.arange(num_episodes)
        episodes = np.arange(num_episodes)

        # Plot the data
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.gca()
        ax2 = ax1.twinx()
        ax1.plot(episodes, rewards, 'g.')
        ax2.plot(episodes, lnt_resets, 'b-')
        ax2.plot(episodes, baseline_resets, 'b--')

        # Label the plot
        ax1.set_ylabel('average step reward', color='g', fontsize=20)
        ax1.tick_params('y', colors='g')
        ax2.set_ylabel('num. resets', color='b', fontsize=20)
        ax2.tick_params('y', colors='b')
        ax1.set_xlabel('num. episodes', fontsize=20)
        plt.savefig(os.path.join(output_dir, 'plot1.png'))

        # plt.show()
        
    def training_iter_callback(self, iter):
        self._training_iter = iter
        # print(f"UPDATED TRAINING ITER: {self._training_iter}")
