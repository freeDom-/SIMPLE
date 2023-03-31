import numpy as np
import gym
import os

import argparse
import time
from shutil import copyfile

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from utils.callbacks import callback_wrapper
from utils.files import reset_logs, reset_models
from utils.register import get_environment
from utils.selfplay import selfplay_wrapper

import config
import utils.selfplay as selfplay

from stable_baselines3.common import logger as sb_logger
logger = sb_logger.configure(config.LOGDIR, ['stdout', 'log'])

def main(args):

  # Raise exception on fp error to prevent NaN and inf values
  np.seterr(all='raise')

  model_dir = os.path.join(config.MODELDIR, args.env_name)

  #if rank == 0:
  try:
    os.makedirs(model_dir)
  except:
    pass
  reset_logs()
  if args.reset:
    reset_models(model_dir)

  if args.debug:
    logger.set_level(config.DEBUG)
  else:
    time.sleep(5)
    logger.set_level(config.INFO)

  logger.info('\nSetting up the selfplay training environment opponents...')
  base_env = get_environment(args.env_name)
  selfplay.max_opponent_models = args.max_opponent_models
  train_env = make_vec_env(lambda: selfplay_wrapper(base_env)(opponent_type = args.opponent_type, verbose = args.verbose, logger = logger), n_envs=args.n_envs)
  eval_env = make_vec_env(lambda: selfplay_wrapper(base_env)(opponent_type = args.opponent_type, verbose = args.verbose, logger = logger), n_envs=1)

  params = {'gamma':args.gamma
    , 'n_steps':args.n_steps
    , 'clip_range':args.clip_range
    , 'clip_range_vf':args.clip_range_vf
    , 'ent_coef':args.ent_coef
    , 'vf_coef':args.vf_coef
    , 'n_epochs':args.n_epochs
    , 'learning_rate':args.learning_rate
    , 'batch_size':args.batch_size
    , 'gae_lambda':args.gae_lambda
    , 'max_grad_norm':args.max_grad_norm
    , 'schedule':'linear'
    , 'verbose':1
    , 'tensorboard_log':config.LOGDIR
  }

  time.sleep(5) # allow time for the base model to be saved out when the environment is created

  if args.mask_invalid_actions:
    base_callback = MaskableEvalCallback
    ppo = MaskablePPO
  else:
    base_callback = EvalCallback
    ppo = PPO

  if args.reset or not os.path.exists(os.path.join(model_dir, 'best_model.zip')):
    logger.info('\nLoading the base PPO agent to train...')
    model = ppo.load(os.path.join(model_dir, 'base.zip'), train_env, **params)
  else:
    logger.info('\nLoading the best_model.zip PPO agent to continue training...')
    model = ppo.load(os.path.join(model_dir, 'best_model.zip'), train_env, **params)

  #Callbacks
  logger.info('\nSetting up the selfplay evaluation environment opponents...')
  eval_freq = max(args.eval_freq // args.n_envs, 1)
  callback_args = {
    'eval_env': eval_env,
    'best_model_save_path' : config.TMPMODELDIR,
    'log_path' : config.LOGDIR,
    'eval_freq' : eval_freq,
    'n_eval_episodes' : args.n_eval_episodes,
    'deterministic' : args.best,
    'render' : True,
    'verbose' : 1
  }
  if args.mask_invalid_actions:
    callback_args['use_masking'] = True

  if args.rules:
    logger.error('\nRule-based agent is currently not supported...')
    return
    '''logger.info('\nSetting up the evaluation environment against the rules-based agent...')
    # Evaluate against a 'rules' agent as well
    eval_actual_callback = EvalCallback(
      eval_env = selfplay_wrapper(base_env)(opponent_type = 'rules', verbose = args.verbose),
      eval_freq=1,
      n_eval_episodes=args.n_eval_episodes,
      deterministic = args.best,
      render = True,
      verbose = 0
    )
    callback_args['callback_on_new_best'] = eval_actual_callback'''
    
  # Evaluate the agent against previous versions
  eval_callback = callback_wrapper(base_callback)(args.opponent_type, args.threshold, args.env_name, **callback_args)

  logger.info('\nSetup complete - commencing learning...\n')

  model.learn(total_timesteps=int(1e8), callback=[eval_callback], reset_num_timesteps=False, tb_log_name=args.tb_log_name)

  train_env.close()
  del train_env


def cli() -> None:
  """Handles argument extraction from CLI and passing to main().
  Note that a separate function is used rather than in __name__ == '__main__'
  to allow unit testing of cli().
  """
  # Setup argparse to show defaults on help
  formatter_class = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(formatter_class=formatter_class)


  parser.add_argument("--reset", "-r", action = 'store_true', default = False
                , help="Start retraining the model from scratch")
  parser.add_argument("--opponent_type", "-o", type = str, default = 'mostly_best'
              , help="best / mostly_best / random / base / rules - the type of opponent to train against")
  parser.add_argument("--debug", "-d", action = 'store_true', default = False
              , help="Debug logging")
  parser.add_argument("--verbose", "-v", action = 'store_true', default = False
              , help="Show observation in debug output")
  parser.add_argument("--rules", "-ru", action = 'store_true', default = False
              , help="Evaluate on a ruled-based agent")
  parser.add_argument("--best", "-b", action = 'store_true', default = False
              , help="Uses best moves when evaluating agent")
  parser.add_argument("--env_name", "-e", type = str, default = 'tictactoe'
              , help="Which gym environment to train in: tictactoe, connect4, sushigo, butterfly, geschenkt, frouge, tafl")
  parser.add_argument("--seed", "-s",  type = int, default = 17
              , help="Random seed")
  parser.add_argument("--mask_invalid_actions", "-m", action = 'store_true', default = False
              , help="Use invalid action masking. Environment needs to implement action_masks method, which returns a boolean array containing the action mask (True means valid action)")
  parser.add_argument("--max_opponent_models", "-max", type = int, default = 1000000
              , help="Limit max opponent models saved in memory to prevent out of memory exceptions for big models. -1 means no limit.")

  parser.add_argument("--n_envs", "-n", type=int, default = 1
            , help="How many environments should be used?")
  parser.add_argument("--eval_freq", "-ef",  type = int, default = 20480
            , help="How many timesteps before the agent is evaluated?")
  parser.add_argument("--n_eval_episodes", "-ne",  type = int, default = 50
            , help="How many episodes should be run to test the agent?")
  parser.add_argument("--threshold", "-t",  type = float, default = 0.1
            , help="What score must the agent achieve during evaluation to 'beat' the previous version?")

  parser.add_argument("--gamma", "-g",  type = float, default = 0.99
            , help="The value of gamma in PPO")
  parser.add_argument("--n_steps", "-ns",  type = int, default = 256
            , help="How many timesteps should each actor contribute to the batch?")
  parser.add_argument("--clip_range", "-c",  type = float, default = 0.2
            , help="The clip paramater in PPO")
  parser.add_argument("--clip_range_vf", "-cvf",  type = float, default = None
            , help="The value function clip paramater in PPO")
  parser.add_argument("--ent_coef", "-ent",  type = float, default = 0.01
            , help="The entropy coefficient in PPO")
  parser.add_argument("--vf_coef", "-vf",  type = float, default = 0.5
            , help="The value function coefficient in PPO")
  parser.add_argument("--gae_lambda", "-gae",  type = float, default = 0.95
            , help="The value of lambda in PPO. Factor for trade-off of bias vs variance for Generalized Advantage Estimator")
  parser.add_argument("--max_grad_norm", "-gn",  type = float, default = 0.5
            , help="The maximum value for gradient clipping in PPO")

  parser.add_argument("--n_epochs", "-oe",  type = int, default = 4
            , help="The number of epoch to train the PPO agent per batch")
  parser.add_argument("--learning_rate", "-lr",  type = float, default = 0.00025
            , help="The step size for the PPO optimiser (Learning Rate Range: 0.003 to 5e-6)")
  parser.add_argument("--batch_size", "-bs",  type = int, default = 32
            , help="The minibatch size in the PPO optimiser")

  parser.add_argument("--tb_log_name", "-tb",  type = str, default = 'tb'
            , help="The name of the run for tensorboard logging")

  # Extract args
  args = parser.parse_args()

  # Enter main
  main(args)
  return


if __name__ == '__main__':
  cli()