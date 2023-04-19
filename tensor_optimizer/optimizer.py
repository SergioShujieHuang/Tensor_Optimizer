from abc import ABC, abstractmethod
from . import TT_tensor
import numpy as np
from . import data_loader


class abstract_tensor_optimizer(ABC):
    def __init__(self, target_tensor):
        self.__target_tensor = target_tensor

    def set_target_tensor(self, target_tensor):
        self.__target_tensor = target_tensor

    def get_target_tensor(self):
        return self.__target_tensor

    @abstractmethod
    def stochastic_TT_SGD(self, sampling_rate, iteration_time, learing_rate):
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
            # 初始化TT_tensor的首部
            if i == 0:
                self.__source_TT_tensor.initial_i_G_tensor(i,
                                                           2,
                                                           self.get_target_tensor().shape[i],
                                                           2 ,
                                                           0)
            # 初始化TT_tensor的尾部
            elif i == self.get_target_tensor().ndim - 1:
                self.__source_TT_tensor.initial_i_G_tensor(i,
                                                           2,
                                                           2,
                                                           self.get_target_tensor().shape[i],
                                                           0)
            # 初始化TT_tensor的中间部分
            else:
                self.__source_TT_tensor.initial_i_G_tensor(i,
                                                           3,
                                                           2,
                                                           self.get_target_tensor().shape[i],
                                                           2)

    def stochastic_TT_SGD(self, sampling_rate, iteration_time, learing_rate):
        # 采样个数
        numbers_of_samples = int(self.get_target_tensor().size * sampling_rate)
        if numbers_of_samples < 1:
            print("sampling_rating is so low that can't sample even one element, please set it again")
            return
        # 生成采样点坐标
        samples_index = []
        # for i in range(self.get_target_tensor().ndim):
        #     i_index = np.random.choice(self.get_target_tensor().shape[i], numbers_of_samples)
        #     if i == 0:
        #         for value in i_index:
        #             samples_index.append([value])
        #     else:
        #         for index, value in enumerate(i_index):
        #             samples_index[index].append(value)
        for i in range(self.get_target_tensor().shape[0]):
            for j in range(self.get_target_tensor().shape[1]):
                for k in range(self.get_target_tensor().shape[2]):
                    samples_index.append([i, j, k])
        print("wwowowowowowowowo")
        print(samples_index)
        for one_iteration in range(iteration_time):
            # 计算梯度
            gradient = {}
            for i in range(self.__source_TT_tensor.get_TT_rank()):
                gradient[i] = np.zeros(self.__source_TT_tensor.get_i_G_tensor(i).shape)

                for sample in samples_index:
                    before_index = 0
                    after_index = i + 1
                    G_before = None
                    G_after = None
                    while before_index < i:
                        if before_index == 0:
                            G_before = self.__source_TT_tensor.get_i_G_tensor(before_index)[sample[before_index], :]
                        else:
                            G_before = np.dot(G_before,
                                              self.__source_TT_tensor.
                                                         get_i_G_tensor(before_index)[:, sample[before_index], :])
                        before_index = before_index + 1
                    while after_index <= self.__source_TT_tensor.get_TT_rank()-1:
                        if G_after is None:
                            if after_index == self.__source_TT_tensor.get_TT_rank()-1:
                                G_after = self.__source_TT_tensor.get_i_G_tensor(after_index)[:, sample[after_index]]
                            else:
                                G_after = self.__source_TT_tensor.get_i_G_tensor(after_index)[:, sample[after_index], :]
                        else:
                            if after_index == self.__source_TT_tensor.get_TT_rank() - 1:
                                G_after = np.dot(G_after, self.__source_TT_tensor.
                                                            get_i_G_tensor(after_index)[:, sample[after_index]])
                            else:
                                G_after = np.dot(G_after, self.__source_TT_tensor.
                                                            get_i_G_tensor(after_index)[:, sample[after_index], :])
                        after_index = after_index + 1
                    if G_before is None and G_after is not None:
                        gradient[i][sample[i], :] += (np.dot(self.__source_TT_tensor.
                                                                      get_i_G_tensor(i)[sample[i], :], G_after)
                                                    - self.get_target_tensor()[tuple(sample)]
                                                    ) * G_after.transpose()
                    elif G_before is not None and G_after is None:
                        gradient[i][:, sample[i]] += (np.dot(G_before, self.__source_TT_tensor.
                                                                                get_i_G_tensor(i)[:, sample[i]])
                                                    - self.get_target_tensor()[tuple(sample)]
                                                    ) * G_before.transpose()
                    else:
                        temp_tensor = np.dot(G_before, self.__source_TT_tensor.
                                                                       get_i_G_tensor(i)[:, sample[i], :])
                        gradient[i][:, sample[i], :] += (np.dot(temp_tensor, G_after)
                                                    - self.get_target_tensor()[tuple(sample)]
                                                    ) * (np.outer(G_after, G_before)).transpose()
            for i in range(self.__source_TT_tensor.get_TT_rank()):
                self.__source_TT_tensor.set_i_G_tensor(i, self.get_source_TT_tensor().get_i_G_tensor(i) -
                                                            learing_rate * gradient[i] * (1 / numbers_of_samples))
            mse = np.sum((self.get_target_tensor() - self.get_source_TT_tensor().get_approximate_tensor()) ** 2) / \
                  np.size(self.get_target_tensor())
            print("asdasdasdasd")
            print(np.size(self.get_target_tensor()))
            print("RMSE: %f" % np.sqrt(mse))





