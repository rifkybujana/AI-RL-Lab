import sys
import os
import random
import json

import gym
import itertools

import numpy as np
import tensorflow as tf

from collections import deque, namedtuple
from torch.utils.tensorboard import SummaryWriter

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
writer = SummaryWriter(f"runs/{param.run_name}")

checkpoint_path = os.path.join(os.getcwd(), "dqn-atari-breakout-tf")

# He weight initialization
winit = tf.variance_scaling_initializer(scale=2) 
# # Xavier weight initialization
# winit = tf.contrib.layers.xavier_initializer()

class QNetwork():
    def __init__(self, scope="QNet", VALID_ACTIONS=[0, 1, 2, 3]):
        self.scope = scope
        self.VALID_ACTIONS = VALID_ACTIONS
        with tf.variable_scope(scope):
            self._build_model()

    def _build_model(self):
        # input placeholders; input is 4 frames of shape 84x84 
        self.tf_X = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # TD
        self.tf_y = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # action
        self.tf_actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        # normalize input
        X = tf.to_float(self.tf_X) / 255.0

        batch_size = tf.shape(self.tf_X)[0]

        if (param.net == 'bigger'):
            # bigger net
            
            # 3 conv layers
            conv1 = tf.contrib.layers.conv2d(X, 32, 8, 4, padding='VALID', activation_fn=tf.nn.relu, weights_initializer=winit)
            conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, padding='VALID', activation_fn=tf.nn.relu, weights_initializer=winit)
            conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, padding='VALID', activation_fn=tf.nn.relu, weights_initializer=winit)

            # fully connected layers
            flattened = tf.contrib.layers.flatten(conv3)
            fc1 = tf.contrib.layers.fully_connected(flattened, 512, activation_fn=tf.nn.relu, weights_initializer=winit)

        elif (param.net == 'smaller'): 
            # smaller net

            # 2 conv layers
            conv1 = tf.contrib.layers.conv2d(X, 16, 8, 4, padding='VALID', activation_fn=tf.nn.relu, weights_initializer=winit)
            conv2 = tf.contrib.layers.conv2d(conv1, 32, 4, 2, padding='VALID', activation_fn=tf.nn.relu, weights_initializer=winit)

            # fully connected layers
            flattened = tf.contrib.layers.flatten(conv2)
            fc1 = tf.contrib.layers.fully_connected(flattened, 256, activation_fn=tf.nn.relu, weights_initializer=winit)

        # Q(s,a)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(self.VALID_ACTIONS), activation_fn=None, weights_initializer=winit)

        action_one_hot = tf.one_hot(self.tf_actions, tf.shape(self.predictions)[1], 1.0, 0.0, name='action_one_hot')
        self.action_predictions = tf.reduce_sum(self.predictions * action_one_hot, reduction_indices=1, name='act_pred')

        if (param.loss == 'L2'):
            # L2 loss
            self.loss = tf.reduce_mean(tf.squared_difference(self.tf_y, self.action_predictions), name='loss')
        elif (param.loss == 'huber'):
            # Huber loss
            self.loss = tf.reduce_mean(huber_loss(self.tf_y-self.action_predictions), name='loss')

        # optimizer 
        #self.optimizer = tf.train.RMSPropOptimizer(learning_rate=param.learning_rate, momentum=0.95, epsilon=0.01)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=param.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, sess, s):
        return sess.run(self.predictions, { self.tf_X: s})

    def update(self, sess, s, a, y):
        feed_dict = { self.tf_X: s, self.tf_y: y, self.tf_actions: a }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

# huber loss
def huber_loss(x):
    condition = tf.abs(x) < 1.0
    output1 = 0.5 * tf.square(x)
    output2 = tf.abs(x) - 0.5
    return tf.where(condition, output1, output2)

# convert raw Atari RGB image of size 210x160x3 into 84x84 grayscale image
class ImageProcess():
    def __init__(self):
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        return sess.run(self.output, { self.input_state: state })

# copy params from qnet1 to qnet2
def copy_model_parameters(sess, qnet1, qnet2):
    q1_params = [t for t in tf.trainable_variables() if t.name.startswith(qnet1.scope)]
    q1_params = sorted(q1_params, key=lambda v: v.name)
    q2_params = [t for t in tf.trainable_variables() if t.name.startswith(qnet2.scope)]
    q2_params = sorted(q2_params, key=lambda v: v.name)

    update_ops = []

    for q1_v, q2_v in zip(q1_params, q2_params):
        op = q2_v.assign(q1_v)
        update_ops.append(op)

    sess.run(update_ops)

# epsilon-greedy
def epsilon_greedy_policy(qnet, num_actions):
    def policy_fn(sess, observation, epsilon):
        if (np.random.rand() < epsilon): 
            # explore: equal probabiities for all actions
            A = np.ones(num_actions, dtype=float) / float(num_actions)
        else:
            # exploit 
            q_values = qnet.predict(sess, np.expand_dims(observation, 0))[0]
            max_Q_action = np.argmax(q_values)
            A = np.zeros(num_actions, dtype=float)
            A[max_Q_action] = 1.0 
        return A
    return policy_fn

# populate replay memory
def populate_replay_mem(
        sess, 
        env, 
        state_processor, 
        replay_memory_init_size, 
        policy, 
        epsilon_start, 
        epsilon_end, 
        epsilon_decay_steps, 
        VALID_ACTIONS, 
        Transition
    ):
    state = env.reset()
    state = state_processor.process(sess, state)
    state = np.stack([state] * 4, axis=2)

    delta_epsilon = (epsilon_start - epsilon_end)/float(epsilon_decay_steps)

    replay_memory = []

    for i in range(replay_memory_init_size):
        epsilon = max(epsilon_start - float(i) * delta_epsilon, epsilon_end)
        action_probs = policy(sess, state, epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        env.render()   
        next_state, reward, done, _ = env.step(VALID_ACTIONS[action])

        next_state = state_processor.process(sess, next_state)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
        replay_memory.append(Transition(state, action, reward, next_state, done))

        if done:
            state = env.reset()
            state = state_processor.process(sess, state)
            state = np.stack([state] * 4, axis=2)
        else:
            state = next_state

    return replay_memory

def make_env(env_id, run_name):
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    # env = gym.wrappers.ResizeObservation(env, (84, 84))
    # env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    return env

# TRAIN

VALID_ACTIONS = [0, 1, 2, 3]

train_or_test = 'train' #'test' #'train'
train_from_scratch = True
start_iter = 0
start_episode = 0
epsilon_start = param.start_e

env = make_env(param.env_id, param.run_name)
print("Action space size: {}".format(env.action_space.n))

observation = env.reset()
print("Observation space shape: {}".format(observation.shape))

writer = SummaryWriter(f"runs/{param.run_name}")

def deep_q_learning(sess, env, q_net, target_net, state_processor, num_episodes, train_or_test='train', train_from_scratch=True,
                    start_iter=0, start_episode=0, replay_memory_size=250000, replay_memory_init_size=50000, update_target_net_every=10000,
                    gamma=0.99, epsilon_start=1.0, epsilon_end=[0.1,0.01], epsilon_decay_steps=[1e6,1e6], batch_size=32):

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    # policy 
    policy = epsilon_greedy_policy(q_net, len(VALID_ACTIONS))

    # populate replay memory
    if (train_or_test == 'train'):
        print("populating replay memory")
        replay_memory = populate_replay_mem(sess, env, state_processor, replay_memory_init_size, policy, epsilon_start, 
                                            epsilon_end[0], epsilon_decay_steps[0], VALID_ACTIONS, Transition)

    # epsilon start
    if (train_or_test == 'train'):
        delta_epsilon1 = (epsilon_start - epsilon_end[0])/float(epsilon_decay_steps[0])     
        delta_epsilon2 = (epsilon_end[0] - epsilon_end[1])/float(epsilon_decay_steps[1])    
        if (train_from_scratch == True):
            epsilon = epsilon_start
        else:
            if (start_iter <= epsilon_decay_steps[0]):
                epsilon = max(epsilon_start - float(start_iter) * delta_epsilon1, epsilon_end[0])
            elif (start_iter > epsilon_decay_steps[0] and start_iter < epsilon_decay_steps[0]+epsilon_decay_steps[1]):
                epsilon = max(epsilon_end[0] - float(start_iter) * delta_epsilon2, epsilon_end[1])
            else:
                epsilon = epsilon_end[1]      
    elif (train_or_test == 'test'):
        epsilon = epsilon_end[1]

    # total number of time steps 
    total_t = start_iter

    for ep in range(start_episode, num_episodes):
        # env reset
        state = env.reset()
        state = state_processor.process(sess, state)
        state = np.stack([state] * 4, axis=2)

        loss = 0.0
        time_steps = 0
        episode_rewards = 0.0

        ale_lives = 5
        info_ale_lives = ale_lives
        steps_in_this_life = 1000000
        num_no_ops_this_life = 0

        while True:
            if (train_or_test == 'train'):
                #epsilon = max(epsilon - delta_epsilon, epsilon_end) 
                if (total_t <= epsilon_decay_steps[0]):
                    epsilon = max(epsilon - delta_epsilon1, epsilon_end[0]) 
                elif (total_t >= epsilon_decay_steps[0] and total_t <= epsilon_decay_steps[0]+epsilon_decay_steps[1]):
                    epsilon = epsilon_end[0] - (epsilon_end[0]-epsilon_end[1]) / float(epsilon_decay_steps[1]) * float(total_t-epsilon_decay_steps[0]) 
                    epsilon = max(epsilon, epsilon_end[1])           
                else:
                    epsilon = epsilon_end[1]

                # update target net
                if total_t % update_target_net_every == 0:
                    copy_model_parameters(sess, q_net, target_net)
                    print("\n copied params from Q net to target net ")

            time_to_fire = False
            if (time_steps == 0 or ale_lives != info_ale_lives):
                # new game or new life 
                steps_in_this_life = 0
                num_no_ops_this_life = np.random.randint(low=0,high=7)
                action_probs = [0.0, 1.0, 0.0, 0.0]  # fire
                time_to_fire = True
                if (ale_lives != info_ale_lives):
                    ale_lives = info_ale_lives
            else:
                action_probs = policy(sess, state, epsilon)

            steps_in_this_life += 1 
            if (steps_in_this_life < num_no_ops_this_life and not time_to_fire):
                # no-op
                action_probs = [1.0, 0.0, 0.0, 0.0] # no-op

            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            env.render()
            next_state_img, reward, done, info = env.step(VALID_ACTIONS[action]) 
            
            info_ale_lives = int(info['ale.lives'])

            # rewards = -1,0,+1 as done in the paper
            #reward = np.sign(reward)

            next_state_img = state_processor.process(sess, next_state_img)

            # state is of size [84,84,4]; next_state_img is of size[84,84]
            #next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
            next_state = np.zeros((84,84,4),dtype=np.uint8)
            next_state[:,:,0] = state[:,:,1] 
            next_state[:,:,1] = state[:,:,2]
            next_state[:,:,2] = state[:,:,3]
            next_state[:,:,3] = next_state_img    


            episode_rewards += reward  
            time_steps += 1

            if (train_or_test == 'train'):
                # if replay memory is full, pop the first element
                if len(replay_memory) == replay_memory_size:
                    replay_memory.pop(0)

                # save transition to replay memory
                # done = True in replay memory for every loss of life 
                if (ale_lives == info_ale_lives):
                    replay_memory.append(Transition(state, action, reward, next_state, done))   
                else:
                    #print('loss of life ')
                    replay_memory.append(Transition(state, action, reward, next_state, True))               

                # sample a minibatch from replay memory
                samples = random.sample(replay_memory, batch_size)
                states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

                # calculate q values and targets 
                q_values_next = target_net.predict(sess, next_states_batch)
                greedy_q = np.amax(q_values_next, axis=1) 
                targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * gamma * greedy_q
                
                # update net 
                if (total_t % 4 == 0):
                    states_batch = np.array(states_batch)
                    loss = q_net.update(sess, states_batch, action_batch, targets_batch)
            
            if done:
                break

            state = next_state
            total_t += 1

        if (train_or_test == 'train'): 
            print('\n Eisode: ', ep, '| time steps: ', time_steps, '| total episode reward: ', episode_rewards, '| total_t: ', total_t, '| epsilon: ', epsilon, '| replay mem size: ', len(replay_memory))
            writer.add_scalar("charts/episodic_return", episode_rewards, ep)
            writer.add_scalar("charts/episodic_length", total_t, ep)
            writer.add_scalar("charts/td_loss", loss, ep)
        elif (train_or_test == 'test'):
            print('\n Eisode: ', ep, '| time steps: ', time_steps, '| total episode reward: ', episode_rewards, '| total_t: ', total_t, '| epsilon: ', epsilon)

    # save model
    saver.save(tf.get_default_session(), checkpoint_path)


tf.reset_default_graph()

# Q and target networks 
q_net = QNetwork(scope="q",VALID_ACTIONS=VALID_ACTIONS)
target_net = QNetwork(scope="target_q", VALID_ACTIONS=VALID_ACTIONS)

# state processor
state_processor = ImageProcess()

# tf saver
saver = tf.train.Saver()

with tf.Session() as sess:
        # load model/ initialize model
        # if ((train_or_test == 'train' and train_from_scratch == False) or train_or_test == 'test'):
        #             latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        #             print("loading model ckpt {}...\n".format(latest_checkpoint))
        #             saver.restore(sess, latest_checkpoint)
        # elif (train_or_test == 'train' and train_from_scratch == True):
        #             sess.run(tf.global_variables_initializer())
        sess.run(tf.global_variables_initializer())

        # run
        deep_q_learning(sess, env, q_net=q_net, target_net=target_net, state_processor=state_processor, num_episodes=25000,
                        train_or_test=train_or_test, train_from_scratch=train_from_scratch, start_iter=start_iter, start_episode=start_episode,
                        replay_memory_size=300000, replay_memory_init_size=5000, update_target_net_every=10000,
                        gamma=0.99, epsilon_start=epsilon_start, epsilon_end=[0.1,0.01], epsilon_decay_steps=[1e6,1e6], batch_size=32)