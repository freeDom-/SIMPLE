import random
import argparse
import os

from stable_baselines3.common.utils import set_random_seed

from utils.files import load_model, write_results
from utils.register import get_environment
from utils.agents import Agent

import config

from stable_baselines3.common import logger as sb_logger
log_dir = os.path.join(config.LOGDIR, 'test')
logger = sb_logger.configure(log_dir, ['stdout', 'log'])

def main(args):

  if args.debug:
    logger.set_level(config.DEBUG)
  else:
    logger.set_level(config.INFO)
    
  #make environment
  env = get_environment(args.env_name)(verbose = args.verbose, manual = args.manual, logger = logger)
  env.seed(args.seed)
  set_random_seed(args.seed)

  total_rewards = {}

  if args.recommend:
    ppo_model = load_model(env, 'best_model.zip')
    ppo_agent = Agent('best_model', ppo_model, logger = logger)
  else:
    ppo_agent = None


  agents = []

  #load the agents
  if len(args.agents) != env.n_players:
    raise Exception(f'{len(args.agents)} players specified but this is a {env.n_players} player game!')

  for i, agent in enumerate(args.agents):
    if agent == 'human':
      agent_obj = Agent('human', logger = logger)
    elif agent == 'rules':
      agent_obj = Agent('rules', logger = logger)
    elif agent == 'base':
      base_model = load_model(env, 'base.zip')
      agent_obj = Agent('base', base_model, logger = logger)
    else:
      ppo_model = load_model(env, f'{agent}.zip')
      agent_obj = Agent(agent, ppo_model, logger = logger)
    agents.append(agent_obj)
    total_rewards[agent_obj.id] = 0
  
  #play games
  logger.info(f'\nPlaying {args.games} games...')
  for game in range(args.games):
    players = agents[:]

    if args.randomise_players:
      random.shuffle(players)

    obs = env.reset()
    done = False
    
    for i, p in enumerate(players):
      logger.debug(f'Player {i+1} = {p.name}')

    while not done:

      current_player = players[env.current_player_num]
      env.render()
      logger.debug(f'\nCurrent player name: {current_player.name}')

      if args.recommend and current_player.name in ['human', 'rules']:
        # show recommendation from last loaded model
        logger.debug(f'\nRecommendation by {ppo_agent.name}:')
        action = ppo_agent.choose_action(env, choose_best_action = True, mask_invalid_actions = True)

      if current_player.name == 'human':
        action = input('\nPlease choose an action: ')
        try:
          # for int actions
          action = int(action)
        except:
          # for MulitDiscrete action input as list TODO
          while True:
            try:
              action = env.parse_action(action)
            except ValueError:
              action = input('\nPlease choose an action: ')
            except:
              action = eval(action)
              break
            else:
              logger.debug(action)
              break
      elif current_player.name == 'rules':
        logger.debug(f'\n{current_player.name} model choices')
        action = current_player.choose_action(env, choose_best_action = False, mask_invalid_actions = True)
      else:
        logger.debug(f'\n{current_player.name} model choices')
        action = current_player.choose_action(env, choose_best_action = args.best, mask_invalid_actions = True)

      obs, reward, done, _ = env.step(action)

      for r, player in zip(reward, players):
        total_rewards[player.id] += r
        player.points += r

      if args.cont:
        input('Press any key to continue')
    
    env.render()

    logger.info(f"Played {game + 1} games: {total_rewards}")

    if args.write_results:
      write_results(players, game, args.games, env.turns_taken)

    for p in players:
      p.points = 0

  env.close()


def cli() -> None:
  """Handles argument extraction from CLI and passing to main().
  Note that a separate function is used rather than in __name__ == '__main__'
  to allow unit testing of cli().
  """
  # Setup argparse to show defaults on help
  formatter_class = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(formatter_class=formatter_class)

  parser.add_argument("--agents","-a", nargs = '+', type=str, default = ['human', 'human']
                , help="Player Agents (human, ppo version)")
  parser.add_argument("--best", "-b", action = 'store_true', default = False
                , help="Make AI agents choose the best move (rather than sampling)")
  parser.add_argument("--games", "-g", type = int, default = 1
                , help="Number of games to play)")
  # parser.add_argument("--n_players", "-n", type = int, default = 3
  #               , help="Number of players in the game (if applicable)")
  parser.add_argument("--debug", "-d",  action = 'store_true', default = False
            , help="Show logs to debug level")
  parser.add_argument("--verbose", "-v",  action = 'store_true', default = False
            , help="Show observation on debug logging")
  parser.add_argument("--manual", "-m",  action = 'store_true', default = False
            , help="Manual update of the game state on step")
  parser.add_argument("--randomise_players", "-r",  action = 'store_true', default = False
            , help="Randomise the player order")
  parser.add_argument("--recommend", "-re",  action = 'store_true', default = False
            , help="Make recommendations on humans turns")
  parser.add_argument("--cont", "-c",  action = 'store_true', default = False
            , help="Pause after each turn to wait for user to continue")
  parser.add_argument("--env_name", "-e",  type = str, default = 'tictactoe'
            , help="Which game to play?")
  parser.add_argument("--write_results", "-w",  action = 'store_true', default = False
            , help="Write results to a file?")
  parser.add_argument("--seed", "-s",  type = int, default = 17
            , help="Random seed")

  # Extract args
  args = parser.parse_args()

  # Enter main
  main(args)
  return


if __name__ == '__main__':
  cli()