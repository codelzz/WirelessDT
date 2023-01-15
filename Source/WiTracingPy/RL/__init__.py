from gymnasium.envs.registration import register

register(
     id="RL/RLTrack-v0",
     entry_point="RL.envs:RLtrackEnv",
     max_episode_steps=300,
)
