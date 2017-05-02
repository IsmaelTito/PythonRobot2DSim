
config_data = {'epuck_error': 0.05, 'reward_error': 0.025, 'epuck_exp': 4, 'reward_exp': 2.5, 'l_reward_weight': 0.5}


def set_parameters(e_error, r_error, e_exp, r_exp, lr_weight):
    config_data['epuck_error'] = e_error
    config_data['reward_error'] = r_error
    config_data['epuck_exp'] = e_exp
    config_data['reward_exp'] = r_exp
    config_data['l_reward_weight'] = lr_weight
