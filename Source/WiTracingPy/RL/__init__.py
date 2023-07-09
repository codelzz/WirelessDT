from gymnasium.envs.registration import register

register(
     id="RL/RLTrack-v0",
     entry_point="RL.envs:RLtrackEnv",
     max_episode_steps=300,
)

register(
     id="RL/RLTrackOffline-v0",
     entry_point="RL.envs:RLtrackOfflineEnv",
     max_episode_steps=300,
)

register(
     id="RL/RLfuse-v0",
     entry_point="RL.envs:RLfuseEnv",
     max_episode_steps=300,
)


register(
     id="RL/RLfuse-abla",
     entry_point="RL.envs:RLfuseAblaEnv",
     max_episode_steps=300,
)