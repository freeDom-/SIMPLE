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
FILTERS = 128


class MaskedCategoricalProbabilityDistribution(CategoricalProbabilityDistribution):
    def __init__(self, logits, mask):
        super(MaskedCategoricalProbabilityDistribution, self).__init__(logits)
        self.masked_logits = tf.add(logits, mask)

    '''def flatparam(self):
        return self.masked_logits'''

    '''def entropy(self):
        a_0 = self.masked_logits - tf.reduce_max(self.masked_logits, axis=-1, keepdims=True)
        exp_a_0 = tf.exp(a_0)
        z_0 = tf.reduce_sum(exp_a_0, axis=-1, keepdims=True)
        p_0 = exp_a_0 / z_0
        return tf.reduce_sum(p_0 * (tf.log(z_0) - a_0), axis=-1)'''

    def sample(self):
        # Gumbel-max trick to sample
        # a categorical distribution (see http://amid.fish/humble-gumbel)
        uniform = tf.random_uniform(tf.shape(self.masked_logits), dtype=self.masked_logits.dtype)
        sample = tf.argmax(self.masked_logits - tf.log(-tf.log(uniform)), axis=-1)
        print(sample)
        tf.Print(sample, [sample])
        return sample


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            obs, legal_actions = split_input(self.processed_obs)
            #obs = tf.Print(obs, [obs], summarize=-1)
            extracted_features = resnet_extractor(obs, **kwargs)

            self._policy = policy_head(extracted_features)
            self._value_fn, self.q_value = value_head(extracted_features)
            
            # Policy masking
            mask = Lambda(lambda x: (1 - x) * -1e8)(legal_actions)
            self.masked_policy = tf.add(self.policy, mask)
            self._proba_distribution  = MaskedCategoricalProbabilityDistribution(self.policy, mask)
        self._setup_init()
        
    def _setup_init(self):
        with tf.variable_scope("output", reuse=True):
            assert self.policy is not None and self.proba_distribution is not None and self.value_fn is not None
            self._action = self.proba_distribution.sample()
            self._deterministic_action = self.proba_distribution.mode()
            self._neglogp = self.proba_distribution.neglogp(self.action)
            self._policy_proba = tf.nn.softmax(self.masked_policy)
            self._value_flat = self.value_fn[:, 0]

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
    actions = tf.concat([tf.unstack(actions, axis=-1)], 0)
    actions = tf.reshape(actions, [-1, ACTION_LAYERS*GRID_SIZE])
    return features, actions

def value_head(y):
    y = convolutional(y, 1, 1)
    y = Flatten()(y)
    y = dense(y, 256, batch_norm = False)
    vf = dense(y, 1, batch_norm = False, activation = 'tanh', name='vf')
    q = dense(y, ACTIONS, batch_norm = False, activation = 'tanh', name='q')
    return vf, q

def policy_head(y):
    y = convolutional(y, 4, 1)
    y = Flatten()(y)
    policy = dense(y, ACTIONS, batch_norm = False, activation = None, name='pi')

    return policy

def resnet_extractor(y, **kwargs):
    y = convolutional(y, FILTERS, KERNEL_SIZE)
    y = residual(y, FILTERS, KERNEL_SIZE)
    y = residual(y, FILTERS, KERNEL_SIZE)
    y = residual(y, FILTERS, KERNEL_SIZE)
    y = residual(y, FILTERS, KERNEL_SIZE)
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
