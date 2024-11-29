

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

from demo import learn_safely

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Let\'s do safe RL with Leave No Trace!')
    env_list = ['cliff-walker']
    parser.add_argument('--env_name', type=str, default='cliff-walker',
                        help=('Name of the environment. The currently '
                              'supported environments are: %s') % env_list)
    # parser.add_argument('--safety_param', type=float, default=0.3,
    #                     help=('Increasing the safety_param from 0 to 1 makes '
    #                           'the agent safer. A reasonable value is 0.3'))
    parser.add_argument('--output_dir', type=str, default='./tmp',
                        help='Folder for storing results')
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Experiment Name')
    

    args = parser.parse_args()
    assert 0 < args.safety_param < 1, 'safety_param should be between 0 and 1.'
    decay_types = ["cosine", "linear", "exponential"]
    for d in decay_types:
        learn_safely(args.env_name, args.safety_param,
                     args.output_dir, args.exp_name + '-' + d, d)