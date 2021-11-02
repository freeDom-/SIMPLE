import tensorflow as tf

from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from utils.register import get_network_arch, get_environment
from utils.selfplay import selfplay_wrapper

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

base_env = get_environment('tafl')
env = selfplay_wrapper(base_env)(opponent_type = 'mostly_best', verbose = False)

policy = get_network_arch('tafl')

model = PPO2(policy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("tafl_ai")

del model # remove to demonstrate saving and loading

model = PPO2.load("tafl_ai")

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()