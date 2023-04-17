import numpy as np


class TT_tensor(object):
    def __init__(self, TT_rank):
        self.__G_tensor = {}
        self.__TT_rank = TT_rank

    def get_TT_rank(self):
        return self.__TT_rank

    def get_i_G_tensor(self, i):
        return self.__G_tensor[i]

    def set_i_G_tensor(self, i, G_tensor):
        self.__G_tensor[i] = G_tensor

    def get_approximate_tensor(self):
        approximate_tensor = self.__G_tensor[0]
        for i in range(self.__TT_rank - 2):
            temp_tensor = np.swapaxes(self.__G_tensor[i+1], 0, 1)
            approximate_tensor = np.dot(approximate_tensor, temp_tensor)
        approximate_tensor = np.dot(approximate_tensor, self.__G_tensor[self.__TT_rank-1])
        return approximate_tensor

    def initial_i_G_tensor(self, i, tensor_rank, x, y, z):
        if tensor_rank == 2:
            self.__G_tensor[i] = np.random.rand(x, y)
        elif tensor_rank == 3:
            self.__G_tensor[i] = np.random.rand(x, y, z)
