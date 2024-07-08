'''register the gym-like envs, see ExampleCode\RL\learn_hover.py'''

from gymnasium.envs.registration import register

register(
    id='attbody-qyhb-v0',
    entry_point='hitsz_qy_hummingbird.envs.rl_att_body:RLattbody',
    kwargs={'gui': True}
)

register(
    id='att-qyhb-v0',
    entry_point='hitsz_qy_hummingbird.envs.rl_attitude:RLatt',
    kwargs={'gui': True}
)

register(
    id='flip-qyhb-v0',
    entry_point='hitsz_qy_hummingbird.envs.rl_flip:RLflip',
    kwargs={'gui': True}
)

register(
    id='escape-qyhb-v0',
    entry_point='hitsz_qy_hummingbird.envs.rl_escape:RLescape',
    kwargs={'gui': True}
)
