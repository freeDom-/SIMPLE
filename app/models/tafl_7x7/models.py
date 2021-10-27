import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.keras.layers import BatchNormalization, Activation, Flatten, Conv2D, Add, Multiply, Dense, Dropout, Lambda

from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.distributions import CategoricalProbabilityDistributionType, CategoricalProbabilityDistribution

ROWS = 7
COLS = 7
GRID_SIZE = COLS * ROWS
ACTIONS_PER_TOKEN = ROWS + COLS

ACTIONS = ROWS * COLS * ACTIONS_PER_TOKEN
FEATURE_LAYERS = 32
ACTION_LAYERS = ACTIONS_PER_TOKEN
KERNEL_SIZE = 3
FILTERS = 120


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            obs, legal_actions = split_input(self.processed_obs)
            #obs = tf.Print(obs, [obs], summarize=-1)
            extracted_features = resnet_extractor(obs, **kwargs)
            #extracted_features = tf.Print(extracted_features, [extracted_features], summarize=-1)
            
            self._policy = policy_head(extracted_features, legal_actions)
            self._value_fn, self.q_value = value_head(extracted_features)
            self._proba_distribution  = CategoricalProbabilityDistribution(self._policy)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


def split_input(obs):
    #features = obs
    features = obs[...,:-ACTION_LAYERS]
    actions = obs[...,-ACTION_LAYERS:]
    #actions = tf.Print(actions, [actions], summarize=-1)
    actions = tf.concat([tf.unstack(actions, axis=-1)], 0)
    #actions = tf.Print(actions, [actions], summarize=-1)
    actions = tf.reshape(actions, [-1, ACTION_LAYERS*GRID_SIZE])
    #actions = tf.Print(actions, [actions], summarize=-1)
    #actions = actions[...,:ACTIONS]
    #actions = tf.Print(actions, [actions], summarize=-1)
    return features, actions

def value_head(y):
    y = convolutional(y, 2, 1)
    y = Flatten()(y)
    y = dense(y, 256, batch_norm = False)
    vf = dense(y, 1, batch_norm = False, activation = 'tanh', name='vf')
    q = dense(y, ACTIONS, batch_norm = False, activation = 'tanh', name='q')
    return vf, q


def policy_head(y, legal_actions):
    y = convolutional(y, 4, 1)
    y = Flatten()(y)
    policy = dense(y, ACTIONS, batch_norm = False, activation = None, name='pi')
    #policy = tf.Print(policy, [policy], summarize=-1)
    # TODO: multiply policy with legal actions, afterwards subtract -1 for illegal actions
    mask = Lambda(lambda x: (1 - x) * -1e1)(legal_actions)
    #mask = tf.Print(mask, [mask], summarize=-1)
    policy = Add()([policy, mask])
    #policy = Multiply()([policy, legal_actions])
    #policy = tf.Print(policy, [policy], summarize=-1)
    return policy


def resnet_extractor(y, **kwargs):
    part_filters = FILTERS // 3

    a = convolutional(y, part_filters, (COLS, KERNEL_SIZE))
    a = residual(a, part_filters, (COLS, KERNEL_SIZE))
    a = residual(a, part_filters, (COLS, KERNEL_SIZE))
    a = residual(a, part_filters, (COLS, KERNEL_SIZE))
    a = residual(a, part_filters, (COLS, KERNEL_SIZE))
    #a = residual(a, part_filters, (COLS, KERNEL_SIZE))
    #a = residual(a, part_filters, (COLS, KERNEL_SIZE))

    b = convolutional(y, part_filters, (KERNEL_SIZE, ROWS))
    b = residual(b, part_filters, (KERNEL_SIZE, ROWS))
    b = residual(b, part_filters, (KERNEL_SIZE, ROWS))
    b = residual(b, part_filters, (KERNEL_SIZE, ROWS))
    b = residual(b, part_filters, (KERNEL_SIZE, ROWS))
    #b = residual(b, part_filters, (KERNEL_SIZE, ROWS))
    #b = residual(b, part_filters, (KERNEL_SIZE, ROWS))

    c = convolutional(y, part_filters, 5)
    c = residual(c, part_filters, 5)
    c = residual(c, part_filters, 5)
    c = residual(c, part_filters, 5)
    c = residual(c, part_filters, 5)
    #c = residual(c, part_filters, KERNEL_SIZE)
    #c = residual(c, part_filters, KERNEL_SIZE)
    
    y = tf.concat([a,b,c], axis=-1)
    #y = residual(y, FILTERS, KERNEL_SIZE)
    return y


def convolutional(y, filters, kernel_size):
    y = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(y)
    y = BatchNormalization(momentum = 0.9)(y)
    y = Activation('relu')(y)
    return y

def residual(y, filters, kernel_size):
    shortcut = y

    y = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(y)
    y = BatchNormalization(momentum = 0.9)(y)
    y = Activation('relu')(y)

    y = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(y)
    y = BatchNormalization(momentum = 0.9)(y)
    y = Add()([shortcut, y])
    y = Activation('relu')(y)

    return y

def dense(y, filters, batch_norm = True, activation = 'relu', name = None):

    if batch_norm or activation:
        y = Dense(filters)(y)
    else:
        y = Dense(filters, name = name)(y)
    
    if batch_norm:
        if activation:
            y = BatchNormalization(momentum = 0.9)(y)
        else:
            y = BatchNormalization(momentum = 0.9, name = name)(y)

    if activation:
        y = Activation(activation, name = name)(y)
    
    return y


