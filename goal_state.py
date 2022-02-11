import numpy as np

def load_goal_state(config):
    suit, domain_task = config.task.split('_', 1)
    domain, task = domain_task.split('_', 1)
    if domain == 'cup':
      domain = 'ball_in_cup'
    goal_state_obs = np.load('./'+domain+'/'+domain+'_'+task+'.npy')
    return {'image':goal_state_obs}