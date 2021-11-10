# docker-compose exec app python3 train.py -r -e butterfly

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np

import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


import argparse
import time
from shutil import copyfile
from mpi4py import MPI

from stable_baselines.ppo2 import PPO2
from stable_baselines.common.callbacks import EvalCallback

from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan, VecNormalize
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines import logger

from utils.callbacks import SelfPlayCallback
from utils.files import reset_logs, reset_models
from utils.register import get_network_arch, get_environment
from utils.selfplay import selfplay_wrapper

import config

def main(args):

  # Raise exception on fp error to prevent NaN and inf values
  np.seterr(all='raise')
  #tf.add_check_numerics_ops()

  rank = MPI.COMM_WORLD.Get_rank()

  model_dir = os.path.join(config.MODELDIR, args.env_name)

  if rank == 0:
    try:
      os.makedirs(model_dir)
    except:
      pass
    reset_logs(model_dir)
    if args.reset:
      reset_models(model_dir)
    logger.configure(config.LOGDIR)
  else:
    logger.configure(format_strs=[])

  if args.debug:
    logger.set_level(config.DEBUG)
  else:
    time.sleep(5)
    logger.set_level(config.INFO)

  workerseed = args.seed + 10000 * MPI.COMM_WORLD.Get_rank()
  set_global_seeds(workerseed)

  logger.info('\nSetting up the selfplay training environment opponents...')
  base_env = get_environment(args.env_name)
  env = selfplay_wrapper(base_env)(opponent_type = args.opponent_type, verbose = args.verbose)
  env.seed(workerseed)
  train_env = make_vec_env(lambda :env, n_envs=args.n_envs)
  eval_env = make_vec_env(lambda :env, n_envs=1)

  # Check for NaNs and Infs
  #env = make_vec_env(lambda: env)
  #env = VecCheckNan(env, raise_exception=True)
  #env = VecNormalize(env) # when training norm_reward = True
  
  CustomPolicy = get_network_arch(args.env_name)

  params = {'gamma':args.gamma
    , 'n_steps':args.n_steps
    , 'cliprange':args.cliprange
    , 'cliprange_vf':args.cliprange_vf
    , 'ent_coef':args.ent_coef
    , 'vf_coef':args.vf_coef
    , 'noptepochs':args.noptepochs
    , 'learning_rate':args.learning_rate
    , 'nminibatches':args.nminibatches
    , 'lam':args.lam
    , 'max_grad_norm':args.max_grad_norm
    , 'schedule':'linear'
    , 'verbose':1
    , 'tensorboard_log':config.LOGDIR
    , 'n_cpu_tf_sess':args.n_cpu_tf_sess
  }

  time.sleep(5) # allow time for the base model to be saved out when the environment is created

  if args.reset or not os.path.exists(os.path.join(model_dir, 'best_model.zip')):
    logger.info('\nLoading the base PPO agent to train...')
    model = PPO2.load(os.path.join(model_dir, 'base.zip'), train_env, **params)
  else:
    logger.info('\nLoading the best_model.zip PPO agent to continue training...')
    model = PPO2.load(os.path.join(model_dir, 'best_model.zip'), train_env, **params)

  #Callbacks
  logger.info('\nSetting up the selfplay evaluation environment opponents...')
  callback_args = {
    'eval_env': eval_env,
    'best_model_save_path' : config.TMPMODELDIR,
    'log_path' : config.LOGDIR,
    'eval_freq' : args.eval_freq,
    'n_eval_episodes' : args.n_eval_episodes,
    'deterministic' : False,
    'render' : True,
    'verbose' : 0
  }

  if args.rules:  
    logger.info('\nSetting up the evaluation environment against the rules-based agent...')
    # Evaluate against a 'rules' agent as well
    eval_actual_callback = EvalCallback(
      eval_env = selfplay_wrapper(base_env)(opponent_type = 'rules', verbose = args.verbose),
      eval_freq=1,
      n_eval_episodes=args.n_eval_episodes,
      deterministic = args.best,
      render = True,
      verbose = 0
    )
    callback_args['callback_on_new_best'] = eval_actual_callback
    
  # Evaluate the agent against previous versions
  eval_callback = SelfPlayCallback(args.opponent_type, args.threshold, args.env_name, **callback_args)

  logger.info('\nSetup complete - commencing learning...\n')

  model.learn(total_timesteps=int(1e7), callback=[eval_callback], reset_num_timesteps = False, tb_log_name="tb")

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
              , help="Uses best moves when evaluating agent against rules-based agent")
  parser.add_argument("--env_name", "-e", type = str, default = 'tictactoe'
              , help="Which gym environment to train in: tictactoe, connect4, sushigo, butterfly, geschenkt, frouge, tafl")
  parser.add_argument("--seed", "-s",  type = int, default = 17
            , help="Random seed")

  parser.add_argument("--n_cpu_tf_sess", "-cpu", type=int, default = None
            , help="How many threads tensor flow operations may use. None means unlimited.")
  parser.add_argument("--n_envs", "-n", type=int, default = 1
            , help="How many environments should be used?")
  parser.add_argument("--eval_freq", "-ef",  type = int, default = 20480
            , help="How many timesteps should each actor contribute before the agent is evaluated?")
  parser.add_argument("--n_eval_episodes", "-ne",  type = int, default = 100
            , help="How many episodes should each actor contribute to the evaluation of the agent?")
  parser.add_argument("--threshold", "-t",  type = float, default = 0.1
            , help="What score must the agent achieve during evaluation to 'beat' the previous version?")

  parser.add_argument("--gamma", "-g",  type = float, default = 0.99
            , help="The value of gamma in PPO")
  parser.add_argument("--n_steps", "-ns",  type = int, default = 1024
            , help="How many timesteps should each actor contribute to the batch?")
  parser.add_argument("--cliprange", "-c",  type = float, default = 0.2
            , help="The clip paramater in PPO")
  parser.add_argument("--cliprange_vf", "-cvf",  type = float, default = None
            , help="The value function clip paramater in PPO")
  parser.add_argument("--ent_coef", "-ent",  type = float, default = 0.01
            , help="The entropy coefficient in PPO")
  parser.add_argument("--vf_coef", "-vf",  type = float, default = 0.5
            , help="The value function coefficient in PPO")
  parser.add_argument("--lam", "-l",  type = float, default = 0.95
            , help="The value of lambda in PPO")
  parser.add_argument("--max_grad_norm", "-gn",  type = float, default = 0.5
            , help="The maximum value for gradient clipping in PPO")

  parser.add_argument("--noptepochs", "-oe",  type = int, default = 4
            , help="The number of epoch to train the PPO agent per batch")
  parser.add_argument("--learning_rate", "-lr",  type = float, default = 0.00025
            , help="The step size for the PPO optimiser")
  parser.add_argument("--nminibatches", "-nm",  type = int, default = 4
            , help="The minibatch size in the PPO optimiser")

  # Extract args
  args = parser.parse_args()

  # Enter main
  main(args)
  return


if __name__ == '__main__':
  cli()