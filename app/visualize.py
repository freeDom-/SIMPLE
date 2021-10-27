import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import argparse

from utils.files import load_model
from utils.register import get_environment

def main(args):
  env = get_environment(args.env_name)(verbose = args.verbose)
  model = load_model(env, f'{args.model}.zip')
  params = model.get_parameter_list()
  print(params)
  for layer in params:
    layer_name = layer[0]
    weights = layer[1]
    if 'conv' not in layer_name:
      continue
    print(layer_name, weights.shape())

def cli() -> None:
  """Handles argument extraction from CLI and passing to main().
  Note that a separate function is used rather than in __name__ == '__main__'
  to allow unit testing of cli().
  """
  # Setup argparse to show defaults on help
  formatter_class = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(formatter_class=formatter_class)

  parser.add_argument("--model","-m", type=str, default = 'base'
            , help="Model version")
  parser.add_argument("--verbose", "-v",  action = 'store_true', default = False
            , help="Show observation on debug logging")
  parser.add_argument("--env_name", "-e",  type = str, default = 'TicTacToe'
            , help="Which game to visualize?")

  # Extract args
  args = parser.parse_args()

  # Enter main
  main(args)
  return


if __name__ == '__main__':
  cli()
