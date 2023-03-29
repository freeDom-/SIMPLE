import os
import sys
import random
import csv
import time
import numpy as np

from shutil import rmtree
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common import logger as sb_logger
from stable_baselines3.common.env_util import make_vec_env

from utils.register import get_network_arch, get_policy_kwargs

import config

logger = sb_logger.configure(config.LOGDIR, ['stdout'])

def write_results(players, game, games, episode_length):
    
    out = {'game': game
    , 'games': games
    , 'episode_length': episode_length
    , 'p1': players[0].name
    , 'p2': players[1].name
    , 'p1_points': players[0].points
    , 'p2_points': np.sum([x.points for x in players[1:]])
    }

    if not os.path.exists(config.RESULTSPATH):
        with open(config.RESULTSPATH,'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=out.keys())
            writer.writeheader()

    with open(config.RESULTSPATH,'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=out.keys())
        writer.writerow(out)


def load_model(env, name):

    filename = os.path.join(config.MODELDIR, env.name, name)
    if os.path.exists(filename):
        logger.info(f'Loading {name}')
        cont = True
        while cont:
            try:
                env = make_vec_env(lambda :env, n_envs=1)
                try:
                    ppo_model = MaskablePPO.load(filename, env=env)
                except ValueError:
                    ppo_model = PPO.load(filename, env=env)
                cont = False
            except Exception as e:
                time.sleep(5)
                print(e)
    
    elif name == 'base.zip':
        cont = True
        while cont:
            try:
                # TODO: Only save model for first environment
                try:
                    ppo_model = MaskablePPO(policy=get_network_arch(env.name), env=env,
                                            policy_kwargs=get_policy_kwargs(env.name))
                except ValueError:
                    ppo_model = PPO(policy=get_network_arch(env.name), env=env,
                                    policy_kwargs=get_policy_kwargs(env.name))
                logger.info(f'Saving base.zip PPO model...')
                ppo_model.save(os.path.join(config.MODELDIR, env.name, 'base.zip'))
                
                # TODO: Load model when running multi environments
                #ppo_model = MaskablePPO.load(os.path.join(config.MODELDIR, env.name, 'base.zip'), env=env)

                cont = False
            except IOError as e:
                sys.exit(f'Check zoo/{env.name}/ exists and read/write permission granted to user')
            except Exception as e:
                logger.error(e)
                time.sleep(2)
                
    else:
        raise Exception(f'\n{filename} not found')
    
    return ppo_model


def load_all_models(env):
    modellist = [f for f in os.listdir(os.path.join(config.MODELDIR, env.name)) if f.startswith("_model")]
    modellist.sort()
    models = [load_model(env, 'base.zip')]
    for model_name in modellist:
        models.append(load_model(env, model_name))
    return models

def load_models(env, n, load_most_recent = False):
    """
    Load max n models in addition to base and best model.

    :param n: Number of models to load. Loads all models if n is -1.
    :param load_most_recent: Whether to load the most recent models instead. Load random models if false.
    :return: Load models
    """
    if n == -1:
        return load_all_models(env)

    modellist = [f for f in os.listdir(os.path.join(config.MODELDIR, env.name)) if f.startswith("_model")]
    modellist.sort(reverse=True)
    best_model = modellist.pop()

    models = [load_model(env, 'base.zip')]

    n = min(n, modellist.len())
    samples = if load_most_recent modellist[:n] else random.sample(modellist, n)
    for model_name in samples:
        models.append(load_model(env, model_name))

    models.append(load_model(env, best_model))
    return models

def get_best_model_name(env_name):
    modellist = [f for f in os.listdir(os.path.join(config.MODELDIR, env_name)) if f.startswith("_model")]
    
    if len(modellist)==0:
        filename = None
    else:
        modellist.sort()
        filename = modellist[-1]
        
    return filename

def get_model_stats(filename):
    if filename is None:
        generation = 0
        timesteps = 0
        best_rules_based = -np.inf
        best_reward = -np.inf
    else:
        stats = filename.split('_')
        generation = int(stats[2])
        best_rules_based = float(stats[3])
        best_reward = float(stats[4])
        timesteps = int(stats[5])
    return generation, timesteps, best_rules_based, best_reward


def reset_logs():
    try:
        filelist = [ f for f in os.listdir(config.LOGDIR) if f not in ['.gitignore']]
        for f in filelist:
            if os.path.isfile(f):  
                os.remove(os.path.join(config.LOGDIR, f))

        for i in range(100):
            if os.path.exists(os.path.join(config.LOGDIR, f'tb_{i}')):
                rmtree(os.path.join(config.LOGDIR, f'tb_{i}'))
        
        open(os.path.join(config.LOGDIR, 'log.txt'), 'a').close()
    
        
    except Exception as e :
        print(e)
        print('Reset logs failed')

def reset_models(model_dir):
    try:
        filelist = [ f for f in os.listdir(model_dir) if f not in ['.gitignore']]
        for f in filelist:
            os.remove(os.path.join(model_dir , f))
    except Exception as e :
        print(e)
        print('Reset models failed')