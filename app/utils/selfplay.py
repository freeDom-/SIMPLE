import os
import numpy as np
import random

from utils.files import load_model, load_models, load_all_models, get_best_model_name
from utils.agents import Agent

import config

from stable_baselines3.common import logger as sb_logger

# TODO: make shared variables thread safe for SubprocVecEnv
opponent_models = []
max_opponents = -1
use_most_recent_opponents = False
best_model_name = ""

def selfplay_wrapper(env):
    class SelfPlayEnv(env):
        # wrapper over the normal single player env, but loads the best self play model
        def __init__(self, opponent_type, verbose, logger = None):
            global opponent_models, max_opponents, best_model_name, use_most_recent_opponents
            super(SelfPlayEnv, self).__init__(verbose)
            self.opponent_type = opponent_type
            if not opponent_models:
                opponent_models = load_models(self, max_opponents)
            best_model_name = get_best_model_name(self.name)
            if logger is None:
                self.logger = sb_logger.make_output_format('stdout', config.LOGDIR, log_suffix='')
            else:
                self.logger = logger

        def setup_opponents(self):
            global opponent_models, max_opponents, best_model_name, use_most_recent_opponents
            if self.opponent_type == 'rules':
                self.opponent_agent = Agent('rules', logger = self.logger)
            else:
                # incremental load of new model
                current_best_model_name = get_best_model_name(self.name)
                if current_best_model_name != best_model_name:
                    if max_opponents == -1 or (len(opponent_models) - 2) < max_opponents:
                        if use_most_recent_opponents:
                            # Delete oldes model
                            del opponent_models[1]
                        else:
                            # Delete a random model except base and last model
                            idx = random.randint(1, len(opponent_models) - 2)
                            del opponent_models[idx]
                        opponent_models.append(load_model(self, best_model_name ))
                    else:
                        opponent_models = load_models(max_opponents)
                    best_model_name = current_best_model_name

                if self.opponent_type == 'random':
                    start = 0
                    end = len(opponent_models) - 1
                    i = random.randint(start, end)
                    self.opponent_agent = Agent('ppo_opponent', opponent_models[i], logger = self.logger) 

                elif self.opponent_type == 'best':
                    self.opponent_agent = Agent('ppo_opponent', opponent_models[-1], logger = self.logger)  

                elif self.opponent_type == 'mostly_best':
                    j = random.uniform(0,1)
                    if j < 0.8:
                        self.opponent_agent = Agent('ppo_opponent', opponent_models[-1], logger = self.logger)  
                    else:
                        start = 0
                        end = len(opponent_models) - 1
                        i = random.randint(start, end)
                        self.opponent_agent = Agent('ppo_opponent', opponent_models[i], logger = self.logger)

                elif self.opponent_type == 'base':
                    self.opponent_agent = Agent('base', opponent_models[0], logger = self.logger)

            self.agent_player_num = np.random.choice(self.n_players)
            self.agents = [self.opponent_agent] * self.n_players
            self.agents[self.agent_player_num] = None
            try:
                #if self.players is defined on the base environment
                self.logger.debug(f'Agent plays as Player {self.players[self.agent_player_num].id}')
            except:
                pass


        def reset(self):
            super(SelfPlayEnv, self).reset()
            self.setup_opponents()

            if self.current_player_num != self.agent_player_num:   
                self.continue_game()

            return self.observation

        @property
        def current_agent(self):
            return self.agents[self.current_player_num]

        def continue_game(self):
            observation = None
            reward = None
            done = None

            while self.current_player_num != self.agent_player_num:
                self.render()
                action = self.current_agent.choose_action(self, choose_best_action = False, mask_invalid_actions = True)
                observation, reward, done, _ = super(SelfPlayEnv, self).step(action)
                self.logger.debug(f'Rewards: {reward}')
                self.logger.debug(f'Done: {done}')
                if done:
                    break

            return observation, reward, done, None


        def step(self, action):
            self.render()
            observation, reward, done, _ = super(SelfPlayEnv, self).step(action)
            self.logger.debug(f'Action played by agent: {action}')
            self.logger.debug(f'Rewards: {reward}')
            self.logger.debug(f'Done: {done}')

            if not done:
                package = self.continue_game()
                if package[0] is not None:
                    observation, reward, done, _ = package


            agent_reward = reward[self.agent_player_num]
            self.logger.debug(f'\nReward To Agent: {agent_reward}')

            if done:
                self.render()

            return observation, agent_reward, done, {} 

    return SelfPlayEnv