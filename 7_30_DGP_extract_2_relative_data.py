import json
import numpy as np
import os
import torch
import tqdm
import math
import numpy as np
import gpytorch
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
import random


from torch.nn import Linear
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.variational import VariationalStrategy,MeanFieldVariationalDistribution,CholeskyVariationalDistribution,LMCVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch import settings
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.likelihoods import Likelihood
from gpytorch.models.exact_gp import GP
from gpytorch.models.approximate_gp import ApproximateGP
from gpytorch.models.gplvm.latent_variable import *
from gpytorch.models.gplvm.bayesian_gplvm import BayesianGPLVM
from matplotlib import pyplot as plt
from tqdm.notebook import trange
from gpytorch.means import ZeroMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior
from matplotlib import pyplot as plt
import argparse

# current_device = torch.cuda.current_device()
# torch.cuda.set_device(current_device)

params = argparse.ArgumentParser()
params.add_argument('-num_search', type=int, default=10, help='iteration time of random searching ')
params.add_argument('-repeat_time', type=int, default=1, help='repeat time of random searching ')
params.add_argument('-cross_split', type=int, default=2, help='number of cross validation split ')
params.add_argument('-params_epoch', type=int, default=100, help=' epoch number of finding parameters ')
# params.add_argument('-optimal_epoch', type=int, default=10, help=' epoch number of optimal parameters ')
# params.add_argument('-num_hidden_dgp_dims', type=int, default=1, help=' the number of hidden layer dimension ')
params.add_argument('-optimizer_lr', type=float, default=0.01, help=' the learning rate of the optimizer ')
params.add_argument('-num_points', type=int, default=30, help=' epoch number of optimal parameters ')

args = params.parse_args()

num_search = args.num_search
repeat_time = args.repeat_time
cross_split = args.cross_split
params_epoch = args.params_epoch
# optimal_epoch = args.optimal_epoch
# num_hidden_dgp_dims = args.num_hidden_dgp_dims
optimizer_lr = args.optimizer_lr
num_points = args.num_points


smoke_test = ('CI' in os.environ)

#####
device = 'cuda' if torch.cuda.is_available() else 'cpu'
predicted_y = torch.tensor([[6.4009e-09, 5.6086e-09, 3.9883e-09, 4.6619e-08, 8.7676e-08, 1.4396e-07,
         7.9122e-09, 5.6902e-09, 1.4396e-07, 3.2572e-09, 2.6264e-09, 7.7735e-09,
         2.5582e-09, 4.2041e-09, 7.6789e-09, 4.8199e-08, 6.6381e-09, 7.8469e-09,
         3.2696e-09, 3.3632e-09, 9.9804e-09, 5.4717e-09, 3.4486e-09, 4.4499e-09,
         2.8932e-09, 1.4396e-07, 3.1600e-09, 1.8062e-08, 1.5611e-07, 1.3861e-08,
         1.4396e-07, 7.6624e-09, 4.3212e-09, 7.0780e-08, 5.3355e-09, 5.6381e-09,
         1.4402e-07, 7.4349e-08, 2.6169e-08, 2.9831e-09, 3.5438e-08, 3.0931e-09,
         1.2537e-08, 9.3870e-09, 1.5688e-07, 2.7652e-09, 2.5221e-09, 1.3692e-08,
         9.8038e-09, 5.3988e-09, 1.1914e-08, 6.4060e-09, 1.4396e-07, 4.1380e-09,
         6.9653e-09, 4.6714e-08, 4.6374e-09, 4.2834e-09, 1.1499e-08, 8.8520e-09,
         1.5269e-08, 3.2890e-09, 5.1281e-09, 6.0242e-09, 8.1884e-09, 5.0448e-09,
         5.0039e-09, 4.8574e-09, 1.2671e-08, 5.5221e-09, 2.4806e-08, 7.4965e-09,
         6.5509e-09, 1.9726e-08, 2.6736e-09, 8.1391e-09, 5.1987e-09, 4.1336e-09,
         4.2242e-09, 2.2434e-08, 5.1057e-08, 5.5558e-09, 7.6877e-09, 5.8007e-09,
         1.0752e-08, 3.5225e-09, 3.2000e-09, 1.7966e-08, 3.7202e-09, 1.3959e-08,
         6.1079e-09, 6.0192e-08, 1.0285e-08, 3.2487e-09, 3.0206e-08, 4.4854e-09,
         3.1967e-09, 5.4731e-09, 3.5336e-08, 4.1448e-09, 7.7039e-09, 6.2064e-09,
         2.6168e-09, 7.5600e-09, 8.2241e-09, 5.9783e-09, 2.3791e-08, 1.0065e-08,
         7.6265e-09, 3.6765e-09, 2.9371e-08, 7.3095e-09, 3.2590e-09, 1.0052e-08,
         5.1945e-09, 5.9964e-09, 5.7384e-09, 4.8452e-09, 2.3569e-08, 5.9409e-09,
         5.0077e-09, 1.4396e-07, 7.7476e-09, 4.6088e-09, 1.6154e-08, 5.1713e-09,
         3.2453e-09, 5.8489e-09, 4.6299e-09, 4.9751e-08, 3.9992e-09, 3.0802e-09,
         5.1837e-09, 2.7772e-09, 1.5122e-07, 1.4258e-08, 2.8600e-09, 3.2367e-08,
         5.0342e-09, 5.3928e-09, 6.7869e-09, 4.9999e-09, 3.3795e-08, 3.1969e-09,
         3.0176e-09, 5.3869e-09, 4.8323e-09, 1.4396e-07, 2.6071e-09, 3.2310e-09,
         5.1756e-09, 5.2102e-09, 3.5889e-09, 1.4402e-07, 2.7187e-09, 6.8507e-08,
         4.3277e-09, 6.8554e-09, 4.0863e-09, 1.5881e-07, 8.2035e-09, 2.4369e-08,
         2.2621e-08, 2.5909e-09, 6.8397e-09, 3.1479e-09, 4.5254e-09, 4.1674e-09,
         2.0683e-08, 7.7300e-09, 3.5042e-09, 1.4720e-08, 1.5547e-07, 8.4094e-08,
         4.9071e-09, 2.1724e-08, 4.9382e-09, 3.2039e-09, 4.7079e-09, 9.7115e-09,
         3.1340e-09, 2.3212e-08, 3.1348e-09, 4.1770e-09, 4.5286e-09, 1.0165e-08,
         1.5779e-07, 3.8129e-08, 6.3743e-09, 1.0928e-08, 7.0932e-09, 3.1508e-09,
         1.1771e-08, 2.5688e-08, 5.4118e-08, 5.0491e-09, 2.8646e-09, 3.8020e-09,
         5.4966e-09, 4.5337e-09, 7.5350e-09, 4.8303e-09, 6.7700e-08, 4.3318e-08,
         4.7247e-09, 3.3000e-09, 1.6160e-08, 5.0512e-09, 7.5969e-09, 5.3454e-09,
         1.5459e-07, 4.5215e-09, 4.1771e-09, 1.4483e-08, 4.0300e-09, 3.6448e-09,
         1.3882e-08, 7.9757e-09, 1.4959e-07, 4.9642e-09, 5.4135e-09, 2.8203e-09,
         4.5710e-08, 6.4709e-08, 5.1412e-09, 2.2743e-08, 5.1592e-09, 5.6012e-09,
         4.7173e-09, 4.0636e-09, 1.0329e-08, 3.2210e-09, 6.1453e-09, 3.7425e-09,
         8.6001e-08, 1.4193e-08, 5.9545e-09, 3.2337e-09, 4.0864e-09, 3.1690e-09,
         2.6778e-09, 7.7501e-08, 3.2386e-09, 1.4811e-07, 5.5310e-09, 2.3638e-08,
         8.0166e-08, 3.0830e-09, 6.0141e-09, 1.0349e-08, 5.3787e-09, 1.4411e-07,
         2.8162e-08, 3.3737e-09, 5.9721e-09, 3.3562e-09, 2.6590e-09, 3.1832e-09,
         5.2762e-08, 1.5383e-07, 4.2591e-09, 1.4643e-07, 3.1374e-09, 4.6926e-08,
         3.0950e-08, 5.2359e-09, 3.9436e-09, 3.6184e-09, 1.1978e-08, 1.5578e-08,
         4.5380e-08, 7.9989e-09, 4.7844e-09, 3.1114e-09, 3.1926e-09, 8.1906e-08,
         7.0942e-09, 2.4503e-08, 5.3117e-09, 5.9930e-09, 4.4908e-09, 2.5909e-09,
         4.4066e-09, 4.5945e-09, 1.4456e-08, 5.0614e-09, 1.5561e-08, 2.9241e-09,
         1.4478e-07, 6.2150e-09, 6.6094e-09, 3.7610e-09, 3.2051e-09, 4.9552e-09,
         8.1092e-09, 1.3862e-08, 6.5721e-09, 1.5272e-07, 2.6060e-09, 3.3267e-09,
         4.1460e-08, 4.3543e-09, 9.9211e-09, 4.3600e-09, 3.0895e-09, 7.5122e-09,
         5.0299e-09, 4.1380e-09, 3.7009e-09, 4.4514e-09, 1.4396e-07, 6.0116e-09,
         1.4396e-07, 1.0706e-08, 7.6590e-09, 7.6297e-09, 1.4400e-07, 6.0279e-09,
         6.3981e-09, 3.2487e-09, 3.3414e-09, 6.4158e-09, 7.8463e-09, 1.2312e-08,
         3.2274e-09, 1.3514e-08]]).to(device)






test = torch.tensor([[7.6223e-09, 6.0378e-09, 1.4139e-09, 3.9698e-08, 6.3721e-09, 8.9679e-10,
         3.4964e-09, 6.4227e-09, 2.2705e-08, 6.9311e-10, 3.1790e-09, 1.6855e-09,
         8.0335e-10, 1.9709e-09, 2.7991e-09, 8.7126e-08, 6.5505e-10, 1.6213e-09,
         4.2696e-10, 1.0260e-09, 2.5600e-09, 3.2827e-09, 1.1212e-08, 1.0657e-09,
         7.6343e-10, 2.4529e-08, 6.1458e-10, 2.6314e-09, 7.7799e-07, 7.9201e-09,
         5.6582e-10, 6.4762e-09, 6.7655e-10, 7.8787e-09, 1.9445e-09, 4.7559e-09,
         2.8215e-08, 3.9000e-08, 1.1070e-09, 2.4257e-09, 4.8578e-09, 1.0669e-09,
         3.8836e-10, 7.9693e-09, 1.6349e-06, 6.7786e-10, 1.5645e-09, 3.0209e-09,
         4.0203e-08, 6.5831e-10, 7.0926e-10, 3.0080e-09, 8.9844e-10, 1.2177e-09,
         1.0453e-09, 1.0899e-07, 2.1895e-09, 6.9108e-10, 4.0058e-09, 7.0042e-10,
         3.5949e-09, 1.3747e-08, 2.4639e-09, 8.2868e-10, 2.5526e-09, 8.4084e-09,
         1.1875e-08, 4.4602e-10, 6.1284e-10, 2.3141e-09, 2.5324e-08, 1.3722e-08,
         4.7346e-10, 7.1476e-09, 4.5262e-10, 1.7480e-08, 9.3349e-10, 7.4181e-09,
         5.5884e-09, 4.9351e-08, 3.5891e-08, 5.5757e-10, 4.4436e-09, 1.2945e-09,
         1.7228e-09, 1.2998e-09, 1.2061e-09, 5.6473e-09, 9.9873e-09, 1.4891e-09,
         1.6739e-09, 2.9192e-08, 6.6604e-09, 6.0618e-10, 2.5229e-08, 2.6546e-09,
         1.7998e-09, 8.8085e-10, 7.4208e-08, 1.3503e-09, 1.1838e-08, 1.3726e-09,
         8.1940e-10, 2.2437e-09, 3.1021e-08, 6.5560e-10, 2.6366e-09, 3.2001e-08,
         1.3818e-08, 1.8751e-09, 8.9942e-09, 1.5724e-08, 7.2804e-10, 1.2914e-09,
         2.4581e-08, 6.1105e-10, 1.6448e-09, 5.6783e-09, 5.2429e-09, 1.5305e-09,
         4.7281e-10, 2.0476e-08, 4.4209e-09, 9.3025e-10, 1.7999e-08, 6.5242e-09,
         8.9636e-09, 4.0309e-09, 7.7945e-10, 4.3482e-09, 1.4635e-09, 1.8879e-09,
         8.5304e-10, 5.3783e-10, 1.4538e-07, 1.1453e-08, 5.7130e-10, 9.8272e-08,
         1.2306e-09, 2.0955e-09, 8.4738e-10, 3.0733e-09, 1.0114e-07, 6.5019e-10,
         4.8507e-10, 1.5151e-09, 1.1391e-08, 5.7227e-09, 6.3129e-10, 6.8304e-10,
         1.0610e-09, 2.0795e-09, 2.4809e-09, 2.3158e-08, 1.2113e-09, 1.2472e-07,
         2.3668e-08, 1.8423e-09, 6.5796e-10, 3.1196e-07, 4.9469e-09, 4.8914e-09,
         2.1826e-08, 1.5060e-09, 8.9857e-09, 6.8095e-09, 5.0558e-10, 5.8749e-10,
         3.7065e-08, 1.6215e-09, 4.7642e-10, 1.7106e-09, 3.8769e-08, 7.5127e-08,
         5.8438e-09, 3.4649e-08, 2.9076e-09, 3.4399e-09, 1.2481e-09, 3.1793e-09,
         6.3967e-10, 4.8448e-08, 1.5616e-09, 2.1662e-09, 2.9079e-09, 7.1862e-09,
         1.7983e-07, 5.0278e-08, 1.1998e-09, 3.5351e-09, 1.4171e-09, 1.3213e-09,
         3.6966e-09, 9.5407e-09, 4.4244e-10, 4.7407e-09, 6.1544e-10, 2.2785e-08,
         3.5973e-09, 1.2021e-09, 2.0340e-09, 1.0026e-09, 1.8125e-08, 3.2856e-09,
         5.2041e-10, 7.7986e-10, 2.4942e-09, 1.2964e-08, 1.0997e-08, 2.7873e-08,
         5.9825e-08, 1.9266e-09, 6.0758e-10, 5.9901e-10, 1.3085e-09, 7.8387e-10,
         1.0523e-08, 2.2724e-08, 7.3374e-07, 2.4156e-09, 1.3502e-09, 1.6388e-09,
         3.7048e-08, 3.0484e-08, 1.1035e-09, 9.9479e-09, 8.1789e-09, 5.7449e-09,
         1.2525e-08, 1.1373e-09, 6.0153e-09, 8.4237e-10, 2.4285e-09, 1.0218e-09,
         9.2155e-09, 3.5202e-08, 3.6836e-09, 5.6378e-10, 1.8533e-09, 3.6224e-09,
         6.9862e-10, 7.6194e-08, 1.5008e-09, 5.9237e-08, 3.0291e-09, 3.0692e-08,
         1.6154e-07, 1.2734e-09, 3.4987e-09, 4.4167e-09, 3.2605e-09, 1.6070e-09,
         1.3112e-07, 2.4269e-09, 1.1213e-09, 1.5599e-09, 1.3831e-09, 6.6904e-09,
         3.2547e-08, 1.1721e-07, 7.7031e-10, 4.2274e-08, 3.1663e-09, 3.2339e-08,
         9.7694e-09, 1.0111e-09, 8.3890e-09, 6.8164e-09, 2.8711e-09, 1.1425e-09,
         6.4917e-09, 6.2641e-09, 1.8312e-09, 8.6507e-10, 8.7073e-10, 1.8199e-08,
         1.0704e-08, 8.0831e-09, 1.5977e-09, 1.4447e-09, 1.4561e-09, 7.3205e-10,
         9.6242e-10, 2.5084e-09, 4.3401e-10, 3.9409e-10, 1.4313e-08, 9.8582e-10,
         4.1158e-08, 2.4038e-09, 4.9838e-09, 1.3505e-09, 2.6977e-09, 8.1286e-09,
         5.3908e-10, 1.3753e-08, 1.0541e-09, 3.1463e-07, 1.9994e-09, 1.4671e-09,
         9.9515e-09, 5.5165e-10, 4.7408e-09, 1.5117e-09, 5.9497e-10, 1.6057e-08,
         3.4760e-10, 5.0810e-10, 2.3181e-09, 9.3925e-10, 4.3661e-09, 2.1711e-08,
         1.4916e-09, 3.9357e-09, 3.8222e-08, 3.6658e-09, 2.4659e-08, 1.0605e-08,
         1.2478e-09, 8.1722e-10, 2.0290e-09, 1.2910e-09, 1.8442e-09, 5.7807e-10,
         2.8115e-09, 1.7934e-09]]).to(device)


# mean = 20 * torch.log10(predicted_y)
# print(mean)
print(test.shape)
print(predicted_y.shape)

#relative error
relative_error = torch.abs((predicted_y-test)) / test
indices = torch.nonzero(relative_error <= 0.5)
# print(indices)
print(indices.shape)
# print(relative_error)
relative_error_np = relative_error.numpy().flatten()
# print(relative_error_np)


indices = torch.nonzero(relative_error<=2)
print('the number of datapoints<2', indices.size(0))
# print('indices', indices)

test = torch.index_select(test, 1, indices[:,1])
# print(test)

torch.manual_seed(8)
data_file = 'april.json'
with open(data_file) as f:
    data = json.load(f)
collect_data = {}
receiver_position = torch.empty(0, dtype=torch.float)
# print(receiver_position)
RSS_value = torch.empty(0)
# print(RSS_value)
for key, value in data.items():
    rx_data = value['rx_data']
    # print(rx_data)
    metadata = value['metadata']
    # print(metadata)
    power = metadata[0]['power']
    # print(power)
    base_station = rx_data[8][3]
    # print(base_station)
    tr_cords = torch.tensor(value["tx_coords"]).float()
    # print(tr_cords)
    if base_station == 'guesthouse-nuc2-b210' and power == 1:
        # 'cbrssdr1-bes-comp',2, 'cbrssdr1-honors-comp', 3,,'cbrssdr1-ustar-comp',5, 'cbrssdr1-hospital-comp',4, 'ebc-nuc1-b210',6,
        # 'guesthouse-nuc2-b210',8,'garage-nuc1-b210',7, 'law73-nuc1-b210', 9
        RSS_sample = torch.tensor([rx_data[8][0]]).float()
        # print(RSS_sample)
        RSS_value = torch.cat((RSS_value, RSS_sample), dim=0)
        # print(RSS)
        # print(RSS.shape)
        receiver_position = torch.cat((receiver_position, tr_cords), dim=0)
        # print(location)


RSS_value = RSS_value.view(RSS_value.size(0),1)#
for i in range(len(receiver_position)):
    receiver_position[i][0] = (receiver_position[i][0] - 40.75) * 1000
    receiver_position[i][1] = (receiver_position[i][1] + 111.83) * 1000
# print('original RSS value',RSS_value)
# print(RSS_value.shape)
shuffle_index = torch.randperm(len(receiver_position))
receiver_position = receiver_position[shuffle_index].to(device)
RSS_value = RSS_value[shuffle_index].to(device)
# print(RSS_value)
# print(RSS_value.shape)
receiver_position = torch.index_select(receiver_position, 0, indices[:,1])
RSS_value = torch.index_select(RSS_value, 0, indices[:,1])
# print('RSS value extract', RSS_value.view(1, RSS_value.size(0)))

train_x = receiver_position[num_points:,:] # cbrssdr1-honors-comp: 409, 359, 309, 189
train_y = RSS_value[num_points:,:]
test_x = receiver_position[0:num_points,:]
test_y = RSS_value[0:num_points,:]
# test_x = train_x
# test_y = train_y

# normalize the train x and test x
mean_norm_x, std_norm_x = train_x.mean(dim=0),train_x.std(dim=0)
train_x = (train_x - mean_norm_x) / (std_norm_x)
test_x = (test_x - mean_norm_x) / (std_norm_x)

# dB domain to decimal domain
train_y = 10 ** (train_y / 10)
test_y = 10 ** (test_y / 10)

# normalize the train y and test y
mean_norm_decimal, std_norm_decimal = train_y.mean(dim=0),train_y.std(dim=0)
train_y = (train_y - mean_norm_decimal) / (std_norm_decimal)
test_y = (test_y - mean_norm_decimal) / (std_norm_decimal)

train_x = train_x.to(device)
train_y = train_y.to(device)
test_x = test_x.to(device)
# test_y = test_y.view(1,test_x.size(0)).to(device)
test_y = test_y.to(device)
# prior_mean = torch.mean(train_y)
# print('prior mean',prior_mean)
# print('decimal train x',train_x)
print('std_norm_decimal',std_norm_decimal)
print('mean_norm_decimal',mean_norm_decimal)
print('decimal train x',train_x.shape)
print('decimal train x',train_y.shape)
print('test y',test_y)
print('test x',test_x)

def initialize_inducing_inputs(X, M):
    kmeans = KMeans(n_clusters=M)
    # print('kmeans',kmeans)
    kmeans.fit(X.cpu())
    # print('kmeans.fit(X)', kmeans.fit(X))
    inducing_inputs = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
    return inducing_inputs
def _init_pca(X, latent_dim):
    U, S, V = torch.pca_lowrank(X, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(X, V[:,:latent_dim]))
def increase_dim(X, latent_dim):
    X = X.cpu().numpy()
    n_samples, n_features = X.shape
    features = [X[:, i] for i in range(n_features)]
    # print('X[:, 0]',X[:, 0])
    # print('X[:, 1]', X[:, 1])
    # print('feature:',features)
    for i in range(2, latent_dim):
        new_feature = (X[:, 0]+X[:, 1]) ** i
        features.append(new_feature)
    X_expanded = np.column_stack(features)
    inducing_inputs=torch.tensor(X_expanded, dtype=torch.float32).to(device)
    return inducing_inputs

# Deep Gaussian Process
class DGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing, linear_mean = False): # 40, 70,100, 160;
        # inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        if input_dims == train_x.shape[-1]:
            inducing_points = torch.empty(0, dtype=torch.float32).to(device)

            for i in range(output_dims):
                inducing_points_i = initialize_inducing_inputs(train_x, num_inducing)
                inducing_points_i = torch.unsqueeze(inducing_points_i, 0)
                # print(inducing_points_i)
                inducing_points = torch.cat((inducing_points, inducing_points_i)).to(torch.float32)
                # print('inducing points for 2', inducing_points)
                # print('inducing point shape2', inducing_points.shape)
        elif input_dims > train_x.shape[-1]:
            inducing_points = torch.empty(0, dtype=torch.float32).to(device)
            for i in range(output_dims):
                inducing_points_i = initialize_inducing_inputs(increase_dim(train_x, input_dims).detach(), num_inducing)
                inducing_points_i = torch.unsqueeze(inducing_points_i, 0)
                # print(inducing_points_i)
                inducing_points = torch.cat((inducing_points, inducing_points_i)).to(torch.float32)
                # print('inducing points for m', inducing_points)
                # print('inducing point shapem', inducing_points.shape)
        else:
            inducing_points = torch.empty(0, dtype=torch.float32).to(device)
            for i in range(output_dims):
                inducing_points_i = initialize_inducing_inputs(_init_pca(train_x, input_dims).detach(), num_inducing)
                inducing_points_i = torch.unsqueeze(inducing_points_i, 0)
                # print(inducing_points_i)
                inducing_points = torch.cat((inducing_points, inducing_points_i)).to(torch.float32)
                # print('inducing points for 2', inducing_points)
        print('inducing points shape', inducing_points.shape)
        batch_shape = torch.Size([output_dims])
        # mean_field variational distribution
        # variational_distribution = MeanFieldVariationalDistribution(
        #     num_inducing_points=num_inducing,
        #     batch_shape=batch_shape
        # )
        # variational_distribution = MeanFieldVariationalDistribution.initialize_variational_distribution()

        # print('variational variational_mean',variational_distribution.variational_mean)
        # print('variational variational_stddev', variational_distribution.variational_stddev)
        # print(variational_distribution.covariance_matrix)

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = ConstantMean() if linear_mean else LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        # include the inducing points
        mean_x =self.mean_module(x)
        # print('mean_x',mean_x)
        covar_x = self.covar_module(x)
        # print('covar_x',covar_x)
        return MultivariateNormal(mean_x, covar_x)

num_tasks = train_y.size(-1)
# num_hidden_dgp_dims = 1
num_hidden_dims_choice = 1
hiden_lengthscale_choice = 0.15
# hiden_lengthscale_choice_2 = [0.25]
hiden_outputscale_choise = 0.8 # [-3.0, -2.0, -1.0] [-0.1, -0.01, 0.1]
last_lengthscale_choice  = 0.05 # last_lengthscale_choice  = [-0.1, 0.1, 0.01, -0.01]
last_outputscale_choise  = 0.5 # -1.5, -1.75, 0.01
likelihood_noise_range   = 0.01

num_inducing_choice = 10


class MultitaskDeepGP(DeepGP):
    def __init__(self, train_x_shape):
        hidden_layer = DGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_hidden_dims_choice,
            num_inducing=num_inducing_choice,
            linear_mean=True
        )

        # second_hidden_layer = DGPHiddenLayer(
        #     input_dims=hidden_layer.output_dims,
        #     output_dims=num_hidden_dgp_dims+1,
        #     linear_mean=True
        # )
        #
        # third_hidden_layer = DGPHiddenLayer(
        #     input_dims=second_hidden_layer.output_dims+train_x_shape[-1],
        #     output_dims=num_hidden_dgp_dims+2,
        #     linear_mean=True
        # )

        last_layer = DGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=num_tasks,
            num_inducing=num_inducing_choice,
            linear_mean=True
        )
        super().__init__()

        self.hidden_layer = hidden_layer
        # self.second_hidden_layer = second_hidden_layer
        # self.third_hidden_layer = third_hidden_layer
        self.last_layer = last_layer

        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, inputs, **kwargs):
        hidden_rep1 = self.hidden_layer(inputs)
        # print('22',inputs.shape)
        # print('11',hidden_rep1)
        # hidden_rep2 = self.second_hidden_layer(hidden_rep1, **kwargs)
        # hidden_rep3 = self.third_hidden_layer(hidden_rep2, inputs, **kwargs)
        output = self.last_layer(hidden_rep1)
        return output

    def predict(self, test_x):
        with torch.no_grad():
            preds = model.likelihood(model(test_x)).to_data_independent_dist()
        return preds.mean.mean(0), preds.variance.mean(0)


model = MultitaskDeepGP(train_x.shape)
if torch.cuda.is_available():
    model = model.cuda()

# hypers = {
#     'hidden_layer.covar_module.base_kernel.raw_lengthscale': torch.tensor([[[-3.0, -2.5]]]).to(device), #-3,
#     'hidden_layer.covar_module.raw_outputscale': torch.tensor([-1.0]).to(device),
#     'last_layer.covar_module.base_kernel.raw_lengthscale':torch.tensor([[[6.0]]]).to(device),
#     'last_layer.covar_module.raw_outputscale': torch.tensor([-0.1]).to(device),  # 0, 1, 0.5
#     'likelihood.raw_task_noises':torch.tensor([0.01]).to(device),
#     'likelihood.raw_noise':torch.tensor([0.01]).to(device),
# }

hypers = {
    'hidden_layer.covar_module.base_kernel.lengthscale': torch.tensor(
        [[[hiden_lengthscale_choice, hiden_lengthscale_choice]]]).to(device),
    # -3,
    'hidden_layer.covar_module.outputscale': torch.tensor(
        [hiden_outputscale_choise]).to(device),
    'last_layer.covar_module.base_kernel.lengthscale': torch.tensor(
        [[[last_lengthscale_choice for i in range(num_hidden_dims_choice)]]]).to(
        device),
    'last_layer.covar_module.outputscale': torch.tensor(
        [last_outputscale_choise]).to(device),  # 0, 1, 0.5
    'likelihood.task_noises': torch.tensor([likelihood_noise_range]).to(device),
    'likelihood.noise': torch.tensor([likelihood_noise_range]).to(device),
}

model.initialize(**hypers)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=optimizer_lr)
mll = DeepApproximateMLL(
    VariationalELBO(model.likelihood, model, num_data=train_y.size(0))).to(device)
# num_epochs = 1 if smoke_test else 100
# num_epochs = 1 if smoke_test else 2000 best
# epochs_iter = tqdm.tqdm(range(params_epoch), desc='Epoch')
training_iter = 2 if smoke_test else params_epoch
loss_set = np.array([])
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f // %.3f   noise: %.3f' % (
    #     i + 1, training_iter, loss.item(),
    #     model.hidden_layer.covar_module.base_kernel.lengthscale.item(),
    #     model.last_layer.covar_module.base_kernel.lengthscale.item(),
    #     model.likelihood.noise.item()
    # ))
    print(f"Iter {i+1}, Loss:{loss.item()}, lengthscale: {model.hidden_layer.covar_module.base_kernel.lengthscale.detach()} // {model.last_layer.covar_module.base_kernel.lengthscale.detach()}, noise:{model.likelihood.noise.detach()}")
    loss_set = np.append(loss_set, loss.item())
    loss.backward()
    optimizer.step()

# test error
model.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions, predictive_variance = model.predict(test_x.float())
    sqrt_covar = predictive_variance.sqrt()

# in decimal
predictions = predictions * std_norm_decimal + mean_norm_decimal
print('std_norm_decimal',std_norm_decimal)
print('mean_norm_decimal',mean_norm_decimal)
test_y = test_y * std_norm_decimal + mean_norm_decimal
sqrt_covar = std_norm_decimal*sqrt_covar
print('print the standard deviation in decimal--test ' , sqrt_covar.view(1, test_x.size(0)))
print('predicted test y in decimal',predictions.view(1,test_x.size(0)))
print('original test data in decimal ',test_y.view(1,test_x.size(0)))
for task in range(0, 1):
    test_rmse = torch.mean(
        torch.pow(predictions[:, task] - test_y[:, task], 2)).sqrt()
    print('. test RMSE: %e ' % test_rmse, ' in decimal')
    max_y = torch.max(test_y[:,task])
    min_y = torch.min(test_y[:,task])
    nrmse = test_rmse / (max_y - min_y)
    print('. test NRMSE: %e ' % nrmse, ' in decimal')
    print('max_y',max_y)
    print('min_y',min_y)

# in dB
predictions = 10 * torch.log10(predictions)
test_y = 10 * torch.log10(test_y)
print('predicted test y in dB',predictions.view(1,test_x.size(0)))
print('original test data in dB',test_y.view(1,test_x.size(0)))


for task in range(0, 1):
    test_rmse = torch.mean(
        torch.pow(predictions[:, task] - test_y[:, task], 2)).sqrt()
    print('. test RMSE: %e ' % test_rmse, ' in dB')
    max_y = torch.max(test_y[:, task])
    min_y = torch.min(test_y[:, task])
    nrmse = test_rmse / (max_y - min_y)
    print('. test NRMSE: %e ' % nrmse, ' in dB')
    print('max_y',max_y)
    print('min_y',min_y)


#train error
model.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions, predictive_variance = model.predict(train_x.float())
    sqrt_covar = predictive_variance.sqrt()

# in decimal
predictions = predictions * std_norm_decimal + mean_norm_decimal
train_y = train_y * std_norm_decimal + mean_norm_decimal
sqrt_covar = std_norm_decimal*sqrt_covar
print('print the standard deviation in decimal--train ' , sqrt_covar.view(1, train_x.size(0)))
print('predicted train y in decimal',predictions[:].view(1,train_x.size(0)))
print('original train data in decimal',train_y.view(1,train_x.size(0)))
for task in range(0, 1):
    train_rmse = torch.mean(
        torch.pow(predictions[:, task] - train_y[:, task], 2)).sqrt()
    print('. train RMSE: %e ' % train_rmse, ' in decimal')
    max_y = torch.max(train_y[:,task])
    min_y = torch.min(train_y[:,task])
    nrmse = train_rmse / (max_y - min_y)
    print('. train NRMSE: %e ' % nrmse, ' in decimal')
    print('max_y',max_y)
    print('min_y',min_y)

# in dB
predictions = 10 * torch.log10(predictions)
train_y = 10 * torch.log10(train_y)
print('predicted train y in dB',predictions[:].view(1,train_x.size(0)))
print('original train data in dB',train_y.view(1,train_x.size(0)))
for task in range(0, 1):
    train_rmse = torch.mean(
        torch.pow(predictions[:, task] - train_y[:, task], 2)).sqrt()
    print('. train RMSE: %e ' % train_rmse, ' in dB')
    max_y = torch.max(train_y[:, task])
    min_y = torch.min(train_y[:, task])
    nrmse = train_rmse / (max_y - min_y)
    print('. train NRMSE: %e ' % nrmse, ' in dB')
    print('max_y', max_y)
    print('min_y', min_y)

print('test y shape',test_y.shape)
print(base_station)


