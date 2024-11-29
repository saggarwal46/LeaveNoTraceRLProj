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
        "min_value": safety_param,
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
    

    args = parser.parse_args()
    assert 0 < args.safety_param < 1, 'safety_param should be between 0 and 1.'
    if args.learn_safely:
        learn_safely(args.env_name, args.safety_param,
                     args.output_dir, args.exp_name, args.decay_type)
    else:
        learn_dangerously(args.env_name, args.safety_param,
                     args.output_dir, args.exp_name)
