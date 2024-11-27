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

import os
import sys
sys.path.append(os.path.join(sys.path[0], 'coach'))
import numpy as np

from coach.agents import DDQNAgent as _DDQNAgent
from coach.agents import DDPGAgent as _DDPGAgent
from coach.configurations import Preset, DQN, DDPG, GymVectorObservation, ExplorationParameters, OUExploration
# from coach. import VisualizationParameters, TaskParameters
from coach.environments.gym_environment_wrapper import GymEnvironmentWrapper
from coach.memories.memory import Transition
from coach.utils import RunPhase
from coach.logger import logger
from lnt import SafetyWrapper


import tensorflow as tf
from tensorboardX import SummaryWriter

def update_tensorboard_mets(self, phase=RunPhase.TRAIN):
        self.summary_writer.add_scalar('Current Episode', self.current_episode, self.training_iteration)
        self.summary_writer.add_scalar('In Heatup', int(phase == RunPhase.HEATUP), self.training_iteration)
        self.summary_writer.add_scalar('ER #Transitions', self.memory.num_transitions(), self.training_iteration)
        self.summary_writer.add_scalar('ER #Episodes', self.memory.length(), self.training_iteration)
        self.summary_writer.add_scalar('Episode Length', self.current_episode_steps_counter, self.training_iteration)
        self.summary_writer.add_scalar('Total steps', self.total_steps_counter, self.training_iteration)
        self.summary_writer.add_scalar('Epsilon', self.exploration_policy.get_control_param(), self.training_iteration)
        if hasattr(self.env.env, "_total_resets"):
            self.summary_writer.add_scalar('Total Resets', self.env.env._total_resets, self.training_iteration)
        if phase == RunPhase.TRAIN:
            self.summary_writer.add_scalar(
                "Training Reward", 
                self.total_reward_in_current_episode, 
                self.training_iteration
            )
        if phase == RunPhase.TEST:
            self.summary_writer.add_scalar(
                'Evaluation Reward', 
                self.total_reward_in_current_episode, 
                self.training_iteration
            )
        
        # Log the signals for Mean, Stdev, Max, Minx
        for signal in self.signals:
            mean = signal.get_mean()
            if mean != "":
                self.summary_writer.add_scalar(f"{signal.name}/Mean", mean, self.training_iteration)
            
            stdev = signal.get_stdev()
            if stdev != "":
                self.summary_writer.add_scalar(f"{signal.name}/Stdev", stdev, self.training_iteration)
            
            max_val = signal.get_max()
            if max_val != "":
                self.summary_writer.add_scalar(f"{signal.name}/Max", max_val, self.training_iteration)
            
            min_val = signal.get_min()
            if min_val != "":
                self.summary_writer.add_scalar(f"{signal.name}/Min", min_val, self.training_iteration)
                
def Agent(env, **kwargs):
    agent_type = kwargs.pop('agent_type')
    if agent_type == 'DDQNAgent':
        return DDQNAgent(env, **kwargs)
    elif agent_type == 'DDPGAgent':
        # raise NotImplementedError('Support for DDPG is not yet implemented')
        return DDPGAgent(env, **kwargs)
    else:
        raise ValueError('Unknown agent_type: %s' % agent_type)


# Overwrite the coach agents to automatically use the default parameters
class DDQNAgent(_DDQNAgent):
    def __init__(self, env, name,  log_dir, num_training_iterations=10000):
        tuning_params = Preset(agent=DQN, env=GymVectorObservation,
                               exploration=ExplorationParameters)
        self.name = name
        tuning_params.sess = tf.Session()
        tuning_params.agent.discount = 0.99
        tuning_params.visualization.dump_csv = True
        tuning_params.num_training_iterations = num_training_iterations
        tuning_params.num_heatup_steps = env._max_episode_steps * tuning_params.batch_size
        tuning_params.exploration.epsilon_decay_steps = 0.66 * num_training_iterations
        env = GymEnvironmentWrapper(tuning_params, env)
        super(DDQNAgent, self).__init__(env, tuning_params, name=name)
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.summary_writer = SummaryWriter(log_dir)
        self.logger.set_dump_dir(log_dir, add_timestamp='True', filename="metrics")
        # print(f"ENV SAFETY WRAPPER {self.name}", isinstance(self.env, SafetyWrapper))



    def get_q(self, obs, action):
        inputs = {'observation': obs[None, :, None]}
        outputs = self.main_network.target_network.predict(inputs)
        return outputs[0, action]
    
    def update_log(self, phase=RunPhase.TRAIN):
        super().update_log(phase)
        update_tensorboard_mets(self, phase)
        if hasattr(self.env.env, "_total_resets"):
            self.logger.create_signal_value('Total Resets',  self.env.env._total_resets)
    
    def tf_writer_close(self):
        """Close the TensorFlow summary writer when done."""
        self.summary_writer.close()


class DDPGAgent(_DDPGAgent):

    def __init__(self, env, name, log_dir, num_training_iterations=1000000):
        self.name = name
        tuning_params = Preset(agent=DDPG(), env=GymVectorObservation,
                               exploration=OUExploration)
        print(name , DDPG.__dict__)
        tuning_params.sess = tf.Session()
        tuning_params.agent.discount = 0.999
        tuning_params.visualization.dump_csv = True
        tuning_params.num_training_iterations = num_training_iterations
        
        
        env = GymEnvironmentWrapper(tuning_params, env)
        super(DDPGAgent, self).__init__(env, tuning_params, name=name)
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.summary_writer = SummaryWriter(log_dir)
        self.logger.set_dump_dir(log_dir, add_timestamp='True', filename="metrics")

    def get_q(self, obs, action):
        inputs = {'observation': obs[None, :, None],
                  'action': action[None, :]}
        outputs = self.main_network.target_network.predict(inputs)
        return outputs[0, 0]
    
    def update_log(self, phase=RunPhase.TRAIN):
        super().update_log(phase)
        update_tensorboard_mets(self, phase)
        if hasattr(self.env.env, "_total_resets"):
            self.logger.create_signal_value('Total Resets',  self.env.env._total_resets)
    
    def tf_writer_close(self):
        """Close the TensorFlow summary writer when done."""
        self.summary_writer.close()
