from policy_iteration_agent import PolicyIterationAgent
from value_iteration_agent import ValueIterationAgent

class AgentFactory(object):
  @staticmethod
  def create_agent(params):
    """Create a type of agent depending on params.
    """

    if params['type'] == 'policy_iteration':
      return PolicyIterationAgent(grid_size=params['grid_size'], 
                                  rewards=params['rewards'], 
                                  transition_matrix=params['transition_matrix'], 
                                  step_func=params['step_func'],
                                  discount=params['discount'])
    elif params['type'] == 'value_iteration':
      return ValueIterationAgent(grid_size=params['grid_size'], 
                                 rewards=params['rewards'], 
                                 transition_matrix=params['transition_matrix'], 
                                 step_func=params['step_func'],
                                 discount=params['discount']) 
    else:
      return None