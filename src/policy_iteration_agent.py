"""Policy iteration agent implementation for the GridWorld.

Andreea Musat, March 2020
"""

from env import GridWorld
import numpy as np
import pygame


class PolicyIterationAgent(object):
  def __init__(self, grid_size, rewards, transition_matrix, step_func, discount):
    """Initialize a Policy Iteration agent.

    Parameters:
    -----------
    grid_size : Integer
      The size of the grid world the agent interacts with.
    rewards : np.array
      Array of floats of shape (grid_size, grid_size) representing the
      reward the agent receives when stepping on a certain cell.
    transition_matrix : np.array
      Array of floats of shape (grid_size, grid_size, 4, 4). 
      transition_matrix[cell_x][cell_y][intended_action][actual_action] = 
      probability that 'intended_action' taken from (cell_x, cell_y) will lead to
      'actual_action' being applied.
    step_func : function
      A deterministic function that takes the action, the current position and the
      grid size and returns the next position after the action is applied. 
    """
    
    self.grid_size = grid_size
    self.rewards = rewards
    self.transition_matrix = transition_matrix
    self.step_func = step_func
    self.discount = discount
    
    # For visualization purposes, keep track of what the agent should do:
    # 'eval-policy', 'improve-policy' or 'nothing' in case of convergence
    # to the optimal policy.
    self.job = 'eval-policy'

    # Arbitrary initalization of the values and policy.
    self.v = np.zeros((grid_size, grid_size), np.float)
    self.pi = np.random.choice(4, (grid_size, grid_size))

  def next_state_value(self, pos, action):
    i, j = pos[0], pos[1]
    probs = self.transition_matrix[i, j, action, :]
    reward = self.rewards[i, j]
    new_v = 0.0
    for a in range(4):
      next_state = self.step_func(a, pos, self.grid_size)
      next_value = self.v[next_state[0], next_state[1]]
      new_v += probs[a] * (reward + self.discount * next_value)
    return new_v

  def policy_evaluation_step(self):
    delta = 0.0
    new_v = np.zeros(self.v.shape, np.float)

    for i in range(self.grid_size):
      for j in range(self.grid_size):
        pos = np.array([i, j])
        action = self.pi[i, j]
        old_v = self.v[i, j]
        new_v[i, j] = self.next_state_value(pos, action)
        delta = max(delta, np.abs(new_v[i, j] - old_v))

    self.v = new_v
    return delta
        
  def policy_improvement_step(self):
    is_stable_policy = True

    for i in range(self.grid_size):
      for j in range(self.grid_size):
        old_action = self.pi[i, j]
        pos = np.array([i, j])
        exp_vals = np.fromiter((self.next_state_value(pos, a) for a in range(4)), float)
        self.pi[i, j] = np.argmax(exp_vals)

        if self.pi[i, j] != old_action:
          is_stable_policy = False

    return is_stable_policy

  def do_job(self):
    """Do one step of policy evaluation or policy improvement.
    """
    print(self.job)

    if self.job == 'eval-policy':
      delta = self.policy_evaluation_step()
      if delta < 1e-7:
        self.job = 'improve-policy'
      print(delta)

    elif self.job == 'improve-policy':
      is_stable_policy = self.policy_improvement_step()
      self.job = 'nothing' if is_stable_policy else 'eval-policy'

  def get_action(self, state):
    return self.pi[state[0], state[1]]


def main():
  grid_size = 10
  grid_world = GridWorld(grid_size, num_obstacles=5, stochastic_cell_ratio=0.1)
  agent = PolicyIterationAgent(grid_size=grid_size, 
                               rewards=grid_world.rewards, 
                               transition_matrix=grid_world.transition_matrix, 
                               step_func=GridWorld.deterministic_step,
                               discount=0.9)

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
        if agent.job == 'nothing':
           action = agent.get_action(grid_world.pos)
           episode_ended, _, _ = grid_world.step(action)

    grid_world.tick_tock()


if __name__ == '__main__':
  main()