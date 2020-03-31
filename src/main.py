"""
Andreea Musat, March 2020
"""

from agent_interface import Agent
from agent_factory import AgentFactory
from env import GridWorld
from policy_iteration_agent import PolicyIterationAgent


def main():
  grid_size = 10
  grid_world = GridWorld(grid_size, num_obstacles=20, stochastic_cell_ratio=0.1)

  params = {}
  params['type'] = 'value_iteration'
  params['grid_size'] = grid_size
  params['rewards'] = grid_world.rewards
  params['transition_matrix'] = grid_world.transition_matrix
  params['step_func'] = GridWorld.deterministic_step
  params['discount'] = 0.9
  agent = AgentFactory.create_agent(params)

  episode_ended = False
  while True:
    grid_world.get_user_input()
    grid_world.draw_with_state_values(agent.v, policy=agent.pi if grid_world.render_policy else None)
    
    if not grid_world.pause:
      if episode_ended:
        grid_world.restart_episode()
        grid_world.draw_black_screen()
        episode_ended = False
      else:
        agent.do_job()
        if agent.ready_to_play():
           action = agent.get_action(grid_world.pos)
           episode_ended, _, _ = grid_world.step(action)

    grid_world.tick_tock()


if __name__ == '__main__':
  main()