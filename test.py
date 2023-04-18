from tensor_optimizer import optimizer
import numpy as np

# create target tensor manually
target_tensor = np.random.rand(20, 10, 15, 20, 17)

test_optimizer = optimizer.TT_tensor_optimizer(target_tensor)
# # 目标TT_tensor
# print(test_optimizer.get_target_tensor())
# print(test_optimizer.get_source_TT_tensor().get_TT_rank())
#
# for i in range(test_optimizer.get_source_TT_tensor().get_TT_rank()):
#     print(test_optimizer.get_source_TT_tensor().get_i_G_tensor(i).shape)
#
# print("approximate tensor:")
# print(test_optimizer.get_source_TT_tensor().get_approximate_tensor().shape)

test_optimizer.stochastic_TT_SGD(sampling_rate=0.000001,
                                 iteration_time=10000,
                                 learing_rate=0.0000001)
