from tensor_optimizer import optimizer
import numpy as np

# create target tensor manually
target_tensor = abs((2 * np.random.randn(5, 5, 5) + 3) // 1)
print(target_tensor)
print((1 / 8) ** ( 1 / 3))
#
test_optimizer = optimizer.TT_tensor_optimizer(target_tensor)
# 目标TT_tensor
print(test_optimizer.get_target_tensor())
print(test_optimizer.get_source_TT_tensor().get_TT_rank())
# #
# for i in range(test_optimizer.get_source_TT_tensor().get_TT_rank()):
#     print(test_optimizer.get_source_TT_tensor().get_i_G_tensor(i).shape)
#
# print("approximate tensor:")
# print(test_optimizer.get_source_TT_tensor().get_approximate_tensor().shape)

sample_rate = 1 / np.size(target_tensor)

test_optimizer.stochastic_TT_SGD(sample_rate,
                                 iteration_time=10000,
                                 learing_rate=0.0001)
