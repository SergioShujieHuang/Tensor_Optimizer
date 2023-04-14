from tensor_optimizer import optimizer
import numpy as np

# create target tensor manually
target_tensor = np.random.rand(20, 10, 15, 20, 14, 29)

test_optimizer = optimizer.TT_tensor_optimizer(target_tensor)
print(test_optimizer.get_target_tensor())
print(test_optimizer.get_source_TT_tensor().get_TT_rank())
