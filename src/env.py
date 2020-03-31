from colors import *
import math
import numpy as np
import pygame


class GridWorld(object):
  def __init__(self, grid_size, num_obstacles, stochastic_cell_ratio, render):
    self.grid_size = grid_size

    # Generate a 'complex enough' environment. Make sure the current
    # and target positions are at least min_dist cells apart.
    min_dist = 5
    while True:
      self.pos = np.random.choice(grid_size, 2)
      self.target = np.random.choice(grid_size, 2)
      if np.abs(self.pos - self.target).sum() >= min_dist:
        break

    # Initialize a zero-reward matrix. The reward is given for being in a
    # state. End state has +10 reward, obstacles have -1 reward, the rest of
    # the cells have 0 reward.
    self.rewards = np.ones((grid_size, grid_size), np.float) * -1.0
    self.rewards[self.target[0]][self.target[1]] = 10

    # Generate some random obstacles with -1 reward.
    isSamePos = lambda p1, p2: (p1 == p2).sum() == grid_size
    self.obstacles = []
    while len(self.obstacles) < num_obstacles:
      obstacle = np.random.choice(grid_size, 2)
      if isSamePos(obstacle, self.pos) or \
         isSamePos(obstacle, self.target):
         continue

      for prev_obstacle in self.obstacles:
        if isSamePos(obstacle, prev_obstacle):
          continue

      self.rewards[obstacle[0]][obstacle[1]] = -10
      self.obstacles.append(obstacle)

    # Add some stochasticity in the environment. For each cell and each action, 
    # there is a small probability of the wind blowing the agent in another 
    # direction than where it wanted to go.
    # 
    # self.transition_matrix[cell_x][cell_y][intended_action][actual_action] = 
    # probability that 'intended_action' taken from (cell_x, cell_y) will lead to
    # 'actual_action' being applied. Only stochastic_cell_ratio will be stochastic.
    self.transition_matrix = np.zeros((self.grid_size, self.grid_size, 4, 4), np.float)
    for i in range(self.grid_size):
      for j in range(self.grid_size):
        if np.random.random() >= stochastic_cell_ratio:
          self.transition_matrix[i, j, :, :] = np.eye(4)
        else:
          # Ugly hardcoded environment stochasticity. Making sure that it's very likely
          # that the intended action will actually applied and the probability of being
          # blown by the wind is small.
          self.transition_matrix[i, j, :, :] = np.random.choice(10, (4, 4)).astype(np.float)
          diagonal = np.random.choice(np.arange(90, 100), (4, ))
          np.fill_diagonal(self.transition_matrix[i, j, :, :], diagonal)
          row_sum = np.sum(self.transition_matrix[i, j, :, :], axis=1)
          self.transition_matrix[i, j, :, :] = (self.transition_matrix[i, j, :, :].T / row_sum).T

    if render:
      self.win_size = self.grid_size * 25
      self.delta = self.win_size // self.grid_size
      self.win = pygame.display.set_mode((self.win_size, self.win_size))
      self.val_to_color = {}

      self.num_colors = 2000
      self.min_val, self.max_val = -1.0, 1.0
      self.colors = linear_gradient("#FF0000", "#00FF00", self.num_colors)

  def restart_episode(self, ):
    self.pos = np.random.choice(self.grid_size, 2)

  @staticmethod
  def deterministic_step(action, pos, grid_size):
    new_pos = np.copy(pos)
    if action == 0:
      new_pos[1] -= 1
    elif action == 2:
      new_pos[1] += 1
    elif action == 1:
      new_pos[0] += 1
    elif action == 3:
      new_pos[0] -= 1

    new_pos = np.maximum(new_pos, np.zeros((2, )))
    new_pos = np.minimum(new_pos, np.ones((2, )) * grid_size - 1)
    new_pos = new_pos.astype(np.int)

    return new_pos

  def step(self, action):
    """ 

    Parameters:
    -----------
    action : Integer
      Where to move the agent, coded as follows: 0 = N, 1 = E, 2 = S, 3 = W.

    Returns:
    --------
    episode_ended : boolean
      True if the agent reached the terminal state with the current step.
    state : dict
      The current state of the agent. TODO TODO 
    reward : np.float
      The reward received in the current state.
    """

    # Account for the stochasticity in the environment. The action that will
    # be applied might be different from what the agent intended to do.
    action = np.random.choice(4, p=self.transition_matrix[self.pos[0]][self.pos[1]][action])
    
    self.pos = GridWorld.deterministic_step(action, self.pos, self.grid_size)
    episode_ended = ((self.pos == self.target).sum() == 2)
    state = {'pos' : self.pos}

    return episode_ended, state, self.rewards[self.pos[0]][self.pos[1]]

  def draw_grid(self):
    x, y = 0, 0
    for i in range(self.grid_size):
      x, y = x + self.delta, y + self.delta
      pygame.draw.line(self.win, white, (x, 0), (x, self.win_size))
      pygame.draw.line(self.win, white, (0, y), (self.win_size, y))

  def draw_square(self, i, j, color, diff=2):
    length = self.delta - diff
    start_x = i * self.delta + diff // 2
    start_y = j * self.delta + diff // 2
    pygame.draw.rect(self.win, color, (start_x, start_y, length, length))

  def draw_circle(self, i, j, color):
    radius = self.delta // 2 - 5
    center_x = (i + 0.5) * self.delta
    center_y = (j + 0.5) * self.delta 
    pygame.draw.circle(self.win, color, (center_x, center_y), radius)
  
  # Source code for this method: 
  # https://stackoverflow.com/questions/56295712/how-to-draw-a-dynamic-arrow-in-pygame
  def draw_arrow(self, lcolor, tricolor, start, end, trirad, thickness=2):
    rad = math.pi/180
    pygame.draw.line(self.win, lcolor, start, end, thickness)
    rotation = (math.atan2(start[1] - end[1], end[0] - start[0])) + math.pi/2
    pygame.draw.polygon(self.win, tricolor, ((end[0] + trirad * math.sin(rotation),
                                        end[1] + trirad * math.cos(rotation)),
                                       (end[0] + trirad * math.sin(rotation - 120*rad),
                                        end[1] + trirad * math.cos(rotation - 120*rad)),
                                       (end[0] + trirad * math.sin(rotation + 120*rad),
                                        end[1] + trirad * math.cos(rotation + 120*rad))))

  def draw_obstacles(self):
    for obstacle in self.obstacles:
      self.draw_square(obstacle[0], obstacle[1], white, 16)

  def draw_current(self, color):
    self.draw_circle(self.pos[0], self.pos[1], color)

  def draw_target(self):
    self.draw_square(self.target[0], self.target[1], blue, 16)

  def draw_policy(self, policy):
    diff = 5
    for i in range(self.grid_size):
      for j in range(self.grid_size):
        start, end = None, None
        if policy[i, j] == 0:
          start = ((i + 0.5) * self.delta, (j + 1) * self.delta - diff)
          end = ((i + 0.5) * self.delta, j * self.delta + diff)

        elif policy[i, j] == 2:
          start = ((i + 0.5) * self.delta, j * self.delta + diff)
          end = ((i + 0.5) * self.delta, (j + 1) * self.delta - diff)
        
        elif policy[i, j] == 1:
          start = (i * self.delta + diff, (j + 0.5) * self.delta)
          end = ((i + 1) * self.delta - diff, (j + 0.5) * self.delta)

        elif policy[i, j] == 3:
          start = ((i + 1) * self.delta - diff, (j + 0.5) * self.delta)
          end = (i * self.delta + diff, (j + 0.5) * self.delta)
        
        if start is not None and end is not None:
          self.draw_arrow(black, black, start, end, 5)

  def draw_black_screen(self):
    self.win.fill(black)

  def draw(self):
    self.win.fill((0, 0, 0))
    self.draw_grid()
    self.draw_obstacles()
    self.draw_current(dark_blue)
    self.draw_target()

  def draw_state_values(self, v, render_policy=False):
    min_val, max_val = np.min(v), np.max(v)
    bucket_size = (max_val - min_val) / self.num_colors
    for i in range(self.grid_size):
      for j in range(self.grid_size):
        # Default color is middle color if all values are the same.
        idx = self.num_colors // 2
        if bucket_size:
          idx = min(self.num_colors - 1, int((v[i][j] - min_val) / bucket_size))
        self.draw_square(i, j, self.colors[idx])

  def draw_with_state_values(self, v, policy=None):
    """
    v has shape (grid_size, grid_size)
    """
    self.draw_state_values(v)
    self.draw_grid()
    self.draw_obstacles()
    self.draw_current(dark_blue)
    self.draw_target()

    if policy is not None:
      self.draw_policy(policy)

  def draw_state_action_values(self, q):
    min_val, max_val = np.min(q), np.max(q)
    bucket_size = (max_val - min_val) / self.num_colors
    for i in range(self.grid_size):
      for j in range(self.grid_size):
        top_y = i * self.delta - 1
        bottom_y = (i + 1) * self.delta + 1
        right_x = (j + 1) * self.delta - 1
        left_x = j * self.delta + 1
        mid_y = (i + 0.5) * self.delta
        mid_x = (j + 0.5) * self.delta

        for k in range(4):
          # Default color is middle color if all values are the same.
          idx = self.num_colors // 2
          if bucket_size:
            idx = min(self.num_colors - 1, int((q[i][j][k] - min_val) / bucket_size))
          
          if k == 0:
            triangle = ((left_x, top_y), (mid_x, mid_y), (right_x, top_y))
          elif k == 1:
            triangle = ((right_x, top_y), (mid_x, mid_y), (right_x, bottom_y))
          elif k == 2:
            triangle = ((right_x, bottom_y), (mid_x, mid_y), (left_x, bottom_y))
          elif k == 3:
            triangle = ((left_x, bottom_y), (mid_x, mid_y), (left_x, top_y))
          
          pygame.draw.polygon(self.win, self.colors[idx], triangle)  

        pygame.draw.line(self.win, white, (left_x - 1, top_y + 1), (right_x + 1, bottom_y - 1))
        pygame.draw.line(self.win, white, (left_x - 1, bottom_y - 1), (right_x + 1, top_y + 1))

  def draw_with_state_action_values(self, q, render_policy=False):
    """ 
    
    Parameters:
    -----------
    q : np.array 
      Array of float (state, action) values of shape (grid_size, grid_size, 4) (N-E-S-W)
    """
    
    self.draw_state_action_values(q)
    self.draw_grid()
    self.draw_obstacles()
    self.draw_current(dark_blue)
    self.draw_target()


def main():
  grid_size = 8
  grid_world = GridWorld(grid_size, num_obstacles=3, stochastic_cell_ratio=0.1, render=True)
  v = np.random.random((grid_size, grid_size)) 
  q = np.random.random((grid_size, grid_size, 4))
  policy = np.random.choice(4, (grid_size, grid_size))

  clock = pygame.time.Clock()
  while True:
    # grid_world.draw()
    grid_world.draw_with_state_values(v, policy=policy)
    # grid_world.draw_with_state_action_values(q)
    grid_world.step(np.random.choice(4))

    pygame.time.delay(50)
    pygame.display.update()
    clock.tick(20)


if __name__ == '__main__':
  main()