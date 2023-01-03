import sys
import os
import random
import json
import base64
import imageio

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym, suite_atari
from tf_agents.environments import tf_py_environment
from tf_agents.environments import batched_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network, network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.networks import categorical_q_network

from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts

class Parameters:
    PATH = os.path.join(os.getcwd(), "DQN", "torch", 'parameter.json')

    def __init__(self):
        with open(self.PATH, "r") as f:
            self.__dict__ = json.loads(f.read())

    def save(self):
        json_data = json.dumps(self, indent=4, default=lambda o: o.__dict__)
        with open(self.PATH, "w") as f:
            f.write(json_data)

param = Parameters()

env = suite_atari.load(
    param.env_id
)