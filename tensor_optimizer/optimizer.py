from abc import ABC, abstractmethod
from . import TT_tensor
from . import data_loader


class abstract_tensor_optimizer(ABC):
    def __init__(self, target_tensor):
        self.__target_tensor = target_tensor
        self.__learning_rate = 0.1

    def get_learning_rate(self):
        return self.__learning_rate

    def set_learning_rate(self, learning_rate):
        self.__learning_rate = learning_rate

    def set_target_tensor(self, target_tensor):
        self.__target_tensor = target_tensor

    def get_target_tensor(self):
        return self.__target_tensor

    @abstractmethod
    def stochastic_TT_SGD(self):
        pass


class TT_tensor_optimizer(abstract_tensor_optimizer):
    def __init__(self, target_tensor):
        super().__init__(target_tensor)
        self.__source_TT_tensor = TT_tensor.TT_tensor(target_tensor.ndim)
        self.__initialize_TT_tensor()

    def get_source_TT_tensor(self):
        return self.__source_TT_tensor

    def __initialize_TT_tensor(self):
        """
        根据target_tensor的各个维度初始化用于近似的TT_tensor
        """
        for i in range(self.get_target_tensor().ndim):
            # 初始化TT_tensor的首部和尾部
            if i == 0:
                self.__source_TT_tensor.initial_i_G_tensor(i, 2, self.get_target_tensor().shape[0], (self.get_target_tensor().shape[0]+self.get_target_tensor().shape[1])/2)

    def stochastic_TT_SGD(self):
        pass
