import os
import numpy as np
from shutil import copyfile

from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

from utils.files import get_best_model_name, get_model_stats

import config

class SelfPlayCallback(EvalCallback):
  def __init__(self, opponent_type, threshold, env_name, *args, **kwargs):
    super(SelfPlayCallback, self).__init__(*args, **kwargs)
    init_callback(self, opponent_type, threshold, env_name)

  def _on_step(self) -> bool:
    if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
      result = super(SelfPlayCallback, self)._on_step() #this will set self.best_mean_reward to the reward from the evaluation as it's previously -np.inf
      callback_step(self, result)
    return True

class MaskableSelfPlayCallback(MaskableEvalCallback):
  def __init__(self, opponent_type, threshold, env_name, *args, **kwargs):
    super(MaskableSelfPlayCallback, self).__init__(*args, **kwargs)
    init_callback(self, opponent_type, threshold, env_name)

  def _on_step(self) -> bool:
    if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
      result = super(MaskableSelfPlayCallback, self)._on_step() #this will set self.best_mean_reward to the reward from the evaluation as it's previously -np.inf
      callback_step(self, result)
    return True


def init_callback(callback, opponent_type, threshold, env_name):
  callback.opponent_type = opponent_type
  callback.model_dir = os.path.join(config.MODELDIR, env_name)
  callback.generation, callback.base_timesteps, pbmr, bmr = get_model_stats(get_best_model_name(env_name))

  #reset best_mean_reward because this is what we use to extract the rewards from the latest evaluation by each agent
  callback.best_mean_reward = -np.inf
  if callback.callback is not None: #if evaling against rules-based agent as well, reset this too
    callback.callback.best_mean_reward = -np.inf

  if callback.opponent_type == 'rules':
    callback.threshold = bmr # the threshold is the overall best evaluation by the agent against a rules-based agent
  else:
    callback.threshold = threshold # the threshold is a constant

def callback_step(callback, result):
  av_reward = callback.last_mean_reward
  total_episodes = np.sum([callback.n_eval_episodes])

  if callback.callback is not None:
    rules_based_rewards = [callback.callback.best_mean_reward]
    av_rules_based_reward = np.mean(rules_based_rewards)

  callback.logger.info("Total episodes ran={}".format(total_episodes))

  #compare the latest reward against the threshold
  if result and av_reward > callback.threshold:
    callback.generation += 1
    callback.logger.info(f"New best model: {callback.generation}\n")

    generation_str = str(callback.generation).zfill(5)
    av_rewards_str = str(round(av_reward,3))

    if callback.callback is not None:
      av_rules_based_reward_str = str(round(av_rules_based_reward,3))
    else:
      av_rules_based_reward_str = str(0)
    
    source_file = os.path.join(config.TMPMODELDIR, f"best_model.zip") # this is constantly being written to - not actually the best model
    target_file = os.path.join(callback.model_dir,  f"_model_{generation_str}_{av_rules_based_reward_str}_{av_rewards_str}_{str(callback.base_timesteps + callback.num_timesteps)}_.zip")
    copyfile(source_file, target_file)
    target_file = os.path.join(callback.model_dir,  f"best_model.zip")
    copyfile(source_file, target_file)

    # if playing against a rules based agent, update the global best reward to the improved metric
    if callback.opponent_type == 'rules':
      callback.threshold  = av_reward
    
  #reset best_mean_reward because this is what we use to extract the rewards from the latest evaluation by each agent
  callback.best_mean_reward = -np.inf

  if callback.callback is not None: #if evaling against rules-based agent as well, reset this too
    callback.callback.best_mean_reward = -np.inf
