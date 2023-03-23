import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import random
import string

from stable_baselines3.common.policies import obs_as_tensor

import config

from stable_baselines3.common import logger as sb_logger
logger = sb_logger.configure(config.LOGDIR, ['stdout'])

def sample_action(action_probs):
    action = np.random.choice(len(action_probs), p = action_probs)
    return action


def mask_actions(legal_actions, action_probs):
    masked_action_probs = np.multiply(legal_actions, action_probs)
    masked_action_probs = masked_action_probs / np.sum(masked_action_probs)
    return masked_action_probs

def action_probability(model, observation):
    obs = model.policy.obs_to_tensor(observation)[0]
    dis = model.policy.get_distribution(obs)
    probs = dis.distribution.probs
    probs_np = np.array(probs.tolist())
    return probs_np

def predict_values(model, observation):
    obs = model.policy.obs_to_tensor(observation)[0]
    values = np.array(model.policy.predict_values(obs).tolist())
    return values

class Agent():
  def __init__(self, name, model = None):
      self.name = name
      self.id = self.name + '_' + ''.join(random.choice(string.ascii_lowercase) for x in range(5))
      self.model = model
      self.points = 0

  def print_top_actions(self, env, action_probs):
    top5_action_idx = np.argsort(action_probs)[:5]
    top5_actions = action_probs[top5_action_idx]
    try:
      logger.debug(f"Top 5 actions: {[str(i) + ' (' + env.parse_action_num(i) + '): ' + str(round(a,2))[:5] for i,a in zip(top5_action_idx, top5_actions)]}")
    except (NotImplementedError, AttributeError) as e:
      logger.debug(f"Top 5 actions: {[str(i) + ': ' + str(round(a,2))[:5] for i,a in zip(top5_action_idx, top5_actions)]}")

  def choose_action(self, env, choose_best_action, mask_invalid_actions):
      if self.name == 'rules':
        action_probs = np.array(env.rules_move())
        value = None
      else:
        # TODO: need verification if first item is current action probs
        action_probs = action_probability(self.model, env.observation)[0]
        # TODO: need verification if first item of tensor is current value
        value = predict_values(self.model, env.observation)[0][0]
        logger.debug(f'Value {value:.2f}')

      self.print_top_actions(env, action_probs)
      
      if mask_invalid_actions:
        action_probs = mask_actions(env.legal_actions, action_probs)
        logger.debug('Masked ->')
        self.print_top_actions(env, action_probs)
        
      action = np.argmax(action_probs)
      try:
        logger.debug(f'Best action {action} ({env.parse_action_num(action)})')
      except (NotImplementedError, AttributeError) as e:
        logger.debug(f'Best action {action}')

      if not choose_best_action:
          action = sample_action(action_probs)
          try:
            logger.debug(f'Sampled action {action} ({env.parse_action_num(action)}) chosen')
          except (NotImplementedError, AttributeError) as e:
            logger.debug(f'Sampled action {action} chosen')

      return action



