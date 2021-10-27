from gym.envs.registration import register

register(
    id='Tafl-v0',
    entry_point='tafl.envs:TaflEnv',
)


