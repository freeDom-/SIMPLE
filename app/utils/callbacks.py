import os
import numpy as np
from shutil import copyfile

from stable_baselines3.common import logger
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from utils.files import get_best_model_name, get_model_stats

import config

class NewBestCallback(BaseCallback):
  def __init__(self, env_name, Üargs, **kwargs):
    super(NewBestCallback, self).__init__(*args, **kwargs)
    self.model_dir = os.path.join(config.MODELDIR, env_name)
    self.generation, self.base_timesteps, pbmr, bmr = get_model_stats(get_best_model_name(env_name))

  def _on_step(self) -> bool:
    self.generation += 1
    # TODO: Get rewards from parent EvalCallback - can locals_ be used?
    self.av_rewards = 0
    self.av_rules_based_reward = 0
    logger.info(f"New best model: {self.generation}\n")

    generation_str = str(self.generation).zfill(5)
    av_rewards_str = str(round(av_reward,3))

    if self.callback is not None:
      av_rules_based_reward_str = str(round(av_rules_based_reward,3))
    else:
      av_rules_based_reward_str = str(0)
    
    # TODO: Check if num_timesteps are enough?
    timesteps = self.base_timesteps + self.num_timesteps

    source_file = os.path.join(config.TMPMODELDIR, f"best_model.zip") # this is constantly being written to - not actually the best model
    target_file = os.path.join(self.model_dir,  f"_model_{generation_str}_{av_rules_based_reward_str}_{av_rewards_str}_{str(timesteps)}_.zip")
    copyfile(source_file, target_file)
    target_file = os.path.join(self.model_dir,  f"best_model.zip")
    copyfile(source_file, target_file)

    # if playing against a rules based agent, update the global best reward to the improved metric
    if self.opponent_type == 'rules':
      self.threshold  = av_reward
    return True

class SelfPlayCallback(EvalCallback):
  def __init__(self, opponent_type, threshold, env_name, *args, **kwargs):
    super(SelfPlayCallback, self).__init__(*args, **kwargs)
    self.opponent_type = opponent_type
    self.model_dir = os.path.join(config.MODELDIR, env_name)
    self.generation, self.base_timesteps, pbmr, bmr = get_model_stats(get_best_model_name(env_name))

    #reset best_mean_reward because this is what we use to extract the rewards from the latest evaluation by each agent
    self.best_mean_reward = -np.inf
    if self.callback is not None: #if evaling against rules-based agent as well, reset this too
      self.callback.best_mean_reward = -np.inf

    if self.opponent_type == 'rules':
      self.threshold = bmr # the threshold is the overall best evaluation by the agent against a rules-based agent
    else:
      self.threshold = threshold # the threshold is a constant


  def _on_step(self) -> bool:

    if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

      result = super(SelfPlayCallback, self)._on_step() #this will set self.best_mean_reward to the reward from the evaluation as it's previously -np.inf

      av_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, deterministic=self.deterministic, render=self.render, reward_threshold=self.threshold, warn=self.warn)
      av_timesteps = np.mean([self.num_timesteps])
      total_episodes = np.sum([self.n_eval_episodes])

      if self.callback is not None:
        rules_based_rewards = [self.callback.best_mean_reward]
        av_rules_based_reward = np.mean(rules_based_rewards)

      logger.info("Eval num_timesteps={}, episode_reward={:.2f} +/- {:.2f}".format(self.num_timesteps, av_reward, std_reward))
      logger.info("Total episodes ran={}".format(total_episodes))

      #compare the latest reward against the threshold
      if result and av_reward > self.threshold:
        self.generation += 1
        logger.info(f"New best model: {self.generation}\n")

        generation_str = str(self.generation).zfill(5)
        av_rewards_str = str(round(av_reward,3))

        if self.callback is not None:
          av_rules_based_reward_str = str(round(av_rules_based_reward,3))
        else:
          av_rules_based_reward_str = str(0)
        
        source_file = os.path.join(config.TMPMODELDIR, f"best_model.zip") # this is constantly being written to - not actually the best model
        target_file = os.path.join(self.model_dir,  f"_model_{generation_str}_{av_rules_based_reward_str}_{av_rewards_str}_{str(self.base_timesteps + self.num_timesteps)}_.zip")
        copyfile(source_file, target_file)
        target_file = os.path.join(self.model_dir,  f"best_model.zip")
        copyfile(source_file, target_file)

        # if playing against a rules based agent, update the global best reward to the improved metric
        if self.opponent_type == 'rules':
          self.threshold  = av_reward
        
      #reset best_mean_reward because this is what we use to extract the rewards from the latest evaluation by each agent
      self.best_mean_reward = -np.inf

      if self.callback is not None: #if evaling against rules-based agent as well, reset this too
        self.callback.best_mean_reward = -np.inf

    return True