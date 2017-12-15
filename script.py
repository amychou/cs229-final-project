'''
Derived from Stanford NMBL's Learning to Run Challenge for
the NIPS 2017 Competition Track.
'''

import opensim as osim
import numpy as np
import sys
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate
from keras.optimizers import Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from osim.env import *
from osim.http.client import Client
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--steps', dest='steps', action='store', default=10000, type=int)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default="example.h5f")
parser.add_argument('--layers', dest='layers', action='store', default=3, type=int)
parser.add_argument('--standing', dest='standing', action='store_true', default=False)
parser.add_argument('--relu', dest='relu', action='store_true', default=False)
parser.add_argument('--actornodes', dest='actornodes', action='store', default=32, type=int)
parser.add_argument('--criticnodes', dest='criticnodes', action='store', default=64, type=int)
args = parser.parse_args()

if args.standing:
    env = StandEnv(args.visualize) # used to train agent to stand
else:
    env = RunEnv(args.visualize)
env.reset()

nb_actions = env.action_space.shape[0]
nallsteps = args.steps
leak_const = 0.2

actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
for i in range(args.layers):
    actor.add(Dense(args.actornodes))
    if args.relu:
        actor.add(Activation('relu'))
    else:
        actor.add(LeakyReLU(alpha=leak_const))
actor.add(Dense(nb_actions))
actor.add(Activation('sigmoid'))

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = concatenate([action_input, flattened_observation])
for i in range(args.layers):
    x = Dense(args.criticnodes)(x)
    if args.relu:
        x = Activation('relu')(x)
    else:
        x = LeakyReLU(alpha=leak_const)(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)

memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=env.noutput)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3,
                  delta_clip=1.)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

if args.train:
    agent.fit(env, nb_steps=nallsteps, visualize=False, verbose=1, nb_max_episode_steps=env.timestep_limit, log_interval=10000)
    agent.save_weights(args.model, overwrite=True)
else:
    agent.load_weights(args.model)
    agent.test(env, nb_episodes=1, visualize=False, nb_max_episode_steps=500)
