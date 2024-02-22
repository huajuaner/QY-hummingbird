from gymnasium.envs.registration import register

register(
    id='hover-qyhb-v0',
    entry_point='hitsz_qy_hummingbird.envs.RL_wrapped:RLMAV',
)