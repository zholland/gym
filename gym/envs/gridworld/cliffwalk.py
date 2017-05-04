import gym
from gym import spaces


class CliffWalkingEnv(gym.Env):
    """
    Implementation of the cliff-walking task from Sutton and Barto
    """

    def __init__(self, x_dim=12, y_dim=4):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.terminal_state = self.x_dim - 1
        self.current_state = 0
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.x_dim*self.y_dim)

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if action == 0:
            return self.__left()
        elif action == 1:
            return self.__up()
        elif action == 2:
            return self.__right()
        elif action == 3:
            return self.__down()

    def _reset(self):
        self.current_state = 0
        return 0

    def __index_to_coords(self, state):
        return int(state % self.x_dim), int(state // self.x_dim)

    def __coords_to_index(self, x, y):
        return y * self.x_dim + x

    def __left(self):
        x, y = self.__index_to_coords(self.current_state)
        if x == 0:
            return self.current_state, -1, self.current_state == self.terminal_state, {}
        else:
            self.current_state = self.__coords_to_index(x - 1, y)
            return self.current_state, -1, self.current_state == self.terminal_state, {}

    def __up(self):
        x, y = self.__index_to_coords(self.current_state)
        if y == self.y_dim - 1:
            return self.current_state, -1, self.current_state == self.terminal_state, {}
        else:
            self.current_state = self.__coords_to_index(x, y + 1)
            return self.current_state, -1, self.current_state == self.terminal_state, {}

    def __right(self):
        x, y = self.__index_to_coords(self.current_state)
        if x == self.x_dim - 1:
            return self.current_state, -1, self.current_state == self.terminal_state, {}
        elif x == 0 and y == 0:
            self.current_state = 0
            return self.current_state, -100, self.current_state == self.terminal_state, {}
        else:
            self.current_state = self.__coords_to_index(x + 1, y)
            return self.current_state, -1, self.current_state == self.terminal_state, {}

    def __down(self):
        x, y = self.__index_to_coords(self.current_state)
        if x == 0:
            if y == 0:
                return self.current_state, -1, self.current_state == self.terminal_state, {}
            else:
                self.current_state = self.__coords_to_index(x, y - 1)
                return self.current_state, -1, self.current_state == self.terminal_state, {}
        elif 0 < x < self.x_dim - 1 and y == 1:
            self.current_state = 0
            return self.current_state, -100, self.current_state == self.terminal_state, {}
        elif x == self.x_dim - 1 and y == 1:  # Terminal state
            self.current_state = self.terminal_state
            return self.current_state, -1, self.current_state == self.terminal_state, {}
        else:
            self.current_state = self.__coords_to_index(x, y - 1)
            return self.current_state, -1, self.current_state == self.terminal_state, {}
