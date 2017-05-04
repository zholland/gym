import gym
from gym import spaces


class WindyGridworldEnv(gym.Env):
    def __init__(self, x_dim=10, y_dim=7):
        """
        wind_strengths is a list of integers >=0 which represents the strengths of the wind at each x position.
        """
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.wind_strengths = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.x_dim*self.y_dim)
        self.terminal_state = self.__coords_to_index(self.x_dim - 3, self.y_dim // 2)  # 7,
        self.current_state = 0

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
        self.current_state = self.__coords_to_index(0, self.y_dim // 2)
        return self.current_state

    def __index_to_coords(self, S):
        return int(S % self.x_dim), int(S // self.x_dim)

    def __coords_to_index(self, x, y):
        return int(y * self.x_dim + x)  #

    def __left(self):
        x, y = self.__index_to_coords(self.current_state)
        new_x = max([x - 1, 0])
        new_y = min([y + self.wind_strengths[x], self.y_dim - 1])
        self.current_state = self.__coords_to_index(new_x, new_y)
        return self.current_state, -1, self.terminal_state == self.current_state, {}

    def __up(self):
        x, y = self.__index_to_coords(self.current_state)
        self.current_state = self.__coords_to_index(x, min([y + 1 + self.wind_strengths[x], self.y_dim - 1]))
        return self.current_state, -1, self.terminal_state == self.current_state, {}

    def __right(self):
        x, y = self.__index_to_coords(self.current_state)
        new_x = min([x + 1, self.x_dim - 1])
        new_y = min([y + self.wind_strengths[x], self.y_dim - 1])
        self.current_state = self.__coords_to_index(new_x, new_y)
        return self.current_state, -1, self.terminal_state == self.current_state, {}

    def __down(self):
        x, y = self.__index_to_coords(self.current_state)
        new_y = max([min([y - 1 + self.wind_strengths[x], self.y_dim - 1]), 0])
        self.current_state = self.__coords_to_index(x, new_y)
        return self.current_state, -1, self.terminal_state == self.current_state, {}
