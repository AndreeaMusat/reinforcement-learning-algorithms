"""
Andreea Musat, March 2020
"""

from agent_interface import Agent
from env import GridWorld
import numpy as np
import pygame

class ValueIterationAgent(Agent):
  def __init__(self, grid_size, rewards, transition_matrix, step_func, discount):
    self.grid_size = grid_size
    self.rewards = rewards
    self.transition_matrix = transition_matrix
    self.step_func = step_func
    self.discount = discount

    self.job = 'improve-value'

    self.v = np.zeros((grid_size, grid_size), np.float)

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

  def value_improvement_step(self):
    delta = 0.0
    new_v = np.zeros(self.v.shape, np.float)

    for i in range(self.grid_size):
      for j in range(self.grid_size):
        pos = np.array([i, j])
        old_v = self.v[i, j]
        new_vs = np.zeros(4, np.float)
        for action in range(4):
          new_vs[action] = self.next_state_value(pos, action)
        action = np.argmax(new_vs)
        new_v[i, j] = new_vs[action]
        delta = max(delta, np.abs(new_v[i, j] - old_v))

    self.v = new_v
    return delta

  def compute_policy(self):
    self.pi = np.zeros((self.grid_size, self.grid_size), np.int)
    for i in range(self.grid_size):
      for j in range(self.grid_size):
        pos = np.array([i, j])
        exp_vals = np.zeros(4, np.float)
        for action in range(4):
          exp_vals[action] = self.next_state_value(pos, action)
        self.pi[i, j] = np.argmax(exp_vals)

  def do_job(self):
    print(self.job)

    if self.job == 'improve-value':
      delta = self.value_improvement_step()
      if delta < 1e-7:
        self.job = 'nothing'
        self.compute_policy()
      print(delta)

  def get_action(self, state):
    return self.pi[state[0], state[1]]

  def ready_to_play(self):
    return self.job == 'nothing'
