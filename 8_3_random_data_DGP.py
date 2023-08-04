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
params.add_argument('-params_epoch', type=int, default=10000, help=' epoch number of finding parameters ')
# params.add_argument('-optimal_epoch', type=int, default=10, help=' epoch number of optimal parameters ')
# params.add_argument('-num_hidden_dgp_dims', type=int, default=1, help=' the number of hidden layer dimension ')
params.add_argument('-optimizer_lr', type=float, default=0.01, help=' the learning rate of the optimizer ')
params.add_argument('-num_points', type=int, default=900, help=' training points boundary ')
params.add_argument('-inducing_value', type=int, default=80, help=' the number of inducing points ')


args = params.parse_args()

num_search = args.num_search
repeat_time = args.repeat_time
cross_split = args.cross_split
params_epoch = args.params_epoch
# optimal_epoch = args.optimal_epoch
# num_hidden_dgp_dims = args.num_hidden_dgp_dims
optimizer_lr = args.optimizer_lr
num_points = args.num_points
inducing_points_value = args.inducing_value

smoke_test = ('CI' in os.environ)

#####
device = 'cuda' if torch.cuda.is_available() else 'cpu'


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# generate the kernel function
def squared_exponential_kernel(x1, x2, output_scale, length_scale):
    pairwise_sq_dists = torch.sum((x1[:, None] - x2) ** 2, dim=-1)

    return output_scale ** 2 * torch.exp(-pairwise_sq_dists / (2 * (length_scale ** 2)))

num_data = 1000
sharp =1
# train x
train_x = np.random.rand(num_data, 2)


# kernel_matrix = np.zeros((num_data, num_data))
length_scale = 0.1
output_scale = 0.01
likelihood_noise = 0.0001

# Kernel
K = squared_exponential_kernel(torch.tensor(train_x), torch.tensor(train_x), output_scale, length_scale)
K += torch.eye(torch.tensor(train_x).size(0)) * likelihood_noise ** 2
K = K.numpy()
print('K',K)
# extract train y from multivariate Gaussian distribution
# mean
mean_vector = np.zeros((num_data,))
# noise_variance = 0.01
train_y = np.random.multivariate_normal(mean_vector, K * sharp).reshape(-1, 1)

receiver_position = torch.tensor(train_x, dtype=torch.float32).to(device)
RSS_value = torch.tensor(train_y, dtype=torch.float32).to(device)
train_x = receiver_position[num_points:,:] # cbrssdr1-honors-comp: 409, 359, 309, 189
# train_x = torch.cat([receiver_position[0:100,:],receiver_position[250:,:]],dim=-2)
# print('train x',train_x)
train_y = RSS_value[num_points:,:]
# train_y = torch.cat([RSS_value[0:100,:],RSS_value[250:,:]])
test_x = receiver_position[0:15,:]
test_y = RSS_value[0:15,:]
# test_x = train_x
# test_y = train_y



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
print('decimal train x',train_x.shape)
# print('decimal train x',train_x)
print('decimal train y',train_y.view(1, train_x.size(0)))

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
hiden_lengthscale_choice = 0.1
# hiden_lengthscale_choice_2 = [0.25]
hiden_outputscale_choise = 0.1 # [-3.0, -2.0, -1.0] [-0.1, -0.01, 0.1]
last_lengthscale_choice  = 2. # last_lengthscale_choice  = [-0.1, 0.1, 0.01, -0.01]
last_outputscale_choise  = 3. # -1.5, -1.75, 0.01
likelihood_noise_range   = 0.1

num_inducing_choice = inducing_points_value


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
    print(f"                 Outputscale: {model.hidden_layer.covar_module.outputscale.detach()} // {model.last_layer.covar_module.outputscale.detach()}")
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





