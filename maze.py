'''
Maze generator
'''
import numpy as np

# TODO: make a real maze
class Maze(object):
  def __init__(self, batch_size, maze_size):
    self.batch_size = batch_size
    self.maze_size = maze_size

  def get_batch(self):
    return self._fake_get_batch()

  def _fake_get_batch(self):
    maze_im = np.random.rand(self.batch_size, self.maze_size, self.maze_size, 1)
    label = np.random.rand(self.batch_size, self.maze_size)
    return maze_im, label

  def dfs(self):
    pass
