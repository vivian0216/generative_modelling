import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random 
import wandb
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from train import train
import math


class Perceptron(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Perceptron, self).__init__()
        #self.flat = nn.Flatten()
        self.dense1 = nn.Linear(in_dim,20)
        self.dense2 = nn.Linear(20,10)
        self.dense3 = nn.Linear(10, out_dim)
    
    def forward(self,x):
        #x = self.flat(x)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x

def mix_gaussian_dataset(N = int, K = int, D = int, mu = None, p = None, cov_type = 'id', stddev = float):
    """
    mu: (K,D) matrix of means, if None random means in [-1,1]^D are sampled
    p: len K array for the probs of sampling from each gaussian, if none its uniform
    cov_type: choose covariance type, id for identity or full for a full (in any case matrix is multiplied by stddev)
    """
    assert cov_type == 'id' or cov_type == 'full'
    assert mu.shape == (K,D)
    #assert len(p) == K

    #y = np.random.randint(0,K, N)
    y = np.random.choice(K, N, True, p)
    x = np.zeros((N, D))

    if p is None:
        p = np.ones(K)/K
    
    if mu is None:
        mu = np.random.uniform(-1, 1, (K,D))
 

    cov = np.zeros((K,D,D))

 
    if cov_type == 'id':
        for i in range(K):
            cov[i] = np.eye(D)*stddev
    
    elif cov_type == 'full':
        temp = np.random.uniform(-1,1,(K,D,D))
        for i in range(K):
            R = np.tril(temp[i], 0)
            cov[i] = np.dot(R,R.T)*stddev


    for i in range(N):
        mu_i = mu[y[i]]
        cov_i = cov[y[i]]
        x[i,:] = np.random.multivariate_normal(mu_i, cov_i, 1)
 
    x = torch.Tensor(x)
    y = torch.Tensor(y).long()

    dataset = [x,y]
    dataset = list(zip(*dataset))

    #print(dataset_probs.shape)

    assert mu.shape == (K,D)
    assert len(p) == K
    assert cov.shape == (K,D,D)

    params = {'num_points': N,
              'dim': D,
              'num_comp': K,
              'means': mu,
              'covariances': cov,
              'mix_coeff': p}

    return dataset, x, params

def plot_dataset(dataset):
    toplot = np.array(list(zip(*dataset))[0])

    plt.scatter(toplot[:,0],toplot[:,1])
    plt.show()

def SGLD_sampler(num_samples = int, D = int , model = nn.Module, steps = int, step_size = float, noise = float):
    samples = torch.rand((num_samples,D)) * 4 - 2
    for _ in range(steps):
        samples.requires_grad_(True)
        samples_logits = model(samples)
        logsumexp = torch.logsumexp(samples_logits, dim=-1)

        # compute gradients
        logsumexp.sum().backward()
        grad = samples.grad
        
        # do step manually
        with torch.no_grad():
            samples = samples + step_size * grad + noise * torch.randn_like(samples)

    return samples

def plot_probs(model = nn.Module, num_points = int, center = list, hw = int, values_type = 'probs', analytical = True, params = dict):
    """
    Plotter, also computes normalizing constant Z (a bit shit idea but saves time)

    center: vector indicating center of the plot
    hw: half width of the edges of squared area to plot
    values_type: choose 'energies' or 'probs'
    analyical: choose if compute probabilities using true densities or learned model density
    """


    assert values_type == 'energies' or values_type == 'probs'

    Z = 0
    x = np.linspace(-hw,hw,num_points)+center[0]
    y = np.linspace(-hw,hw,num_points)+center[1]
    xv, yv = np.meshgrid(x,y, indexing = 'xy')
    

    xv1 = np.ravel(xv)
    yv1 = np.ravel(yv)
    grid = torch.from_numpy(np.stack((xv1, yv1), axis = 1)).float()

    if analytical == False:
        grid_logits = model(grid)
        values = -torch.logsumexp(grid_logits.detach(), dim=-1)

        if values_type == 'probs':
            values = torch.exp(-values)
            h = 2*hw/num_points
            Z = torch.sum(values*h)
            values = values/Z
            print(torch.sum(values*h))

    else:

        values = compute_probs(grid, params)


    values = np.reshape(values, (num_points,num_points))
    levels = np.linspace(values.min(), values.max(), 40)

    fig, ax = plt.subplots()
    cs = ax.contourf(xv, yv, values, levels = levels)

    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel(values_type)

    plt.show()

    return Z

def compute_probs(points, params):
    K = params['num_comp']
    D = params['dim']
    cov = params['covariances']
    mu = params['means']
    p = params['mix_coeff']
    prec = np.zeros((K,D,D))
    for k in range(K):
        prec[k] = np.linalg.inv(cov[k])

    ext_x = np.repeat(points[:,np.newaxis,:], K, axis = 1)

    temp = np.einsum('ijk, jkl, ijl -> ij', ext_x-mu, prec, ext_x - mu)
    temp = np.divide(np.exp(temp/-2),np.sqrt(np.linalg.det(cov)*(2*math.pi)**D))
    points_probs = np.sum(np.multiply(temp, p), axis = 1)
    
    return points_probs

def compute_KL(data, model, params, Z):
    px = compute_probs(data, params)

    temp = model(data)
    temp = -torch.logsumexp(temp.detach(), dim=-1)
    temp = torch.exp(-temp)
    qx = temp/Z

    kl = np.average(np.log(np.divide(px,qx)))
    return kl





N = 10000 #number of points
D = 2 #dimension of the dataset
K = 5 #nr of components
stddev = 0.01
mu = np.array([[-0.5,-0.5],[-0.5,0.5],[0.5,-0.5],[0.5,0.5],[1,1]])
p = None
#p = np.array([0,0,0,0,1])

full_dataset, x, params = mix_gaussian_dataset(N, K, D, mu = mu, p = p, cov_type = 'id', stddev = stddev)


model = Perceptron(in_dim = D, out_dim = K)
"""
cfg = dict(
        step_size = 0.001,
        noise = 0.01, 
        buffer_size = 1000,
        steps = 20,
        reinit_freq = 0.05,
        epochs = 2,
        batch_size = 50,
        learning_rate = 1e-2,
        gen_weight = 1,
        learning_rate_decay = 0.3, #multiply lr after number of epochs specified by learning_rate_epochs
        learning_rate_epochs = 50, #after how many epochs update the lr
        data_fraction = 1,
        train_fraction = 0.8,
    )
        
# Use 10% of the dataset
subset_size = int(len(full_dataset) * cfg.pop('data_fraction'))
indices = np.random.RandomState(seed=42).permutation(len(full_dataset))[:subset_size]
train_split = int(len(indices) * cfg.pop('train_fraction'))
train_indices = indices[:train_split]
test_indices = indices[train_split:]    
train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
wandb.init(project="generative-modelling", config=cfg)

train(model,
      train_dataset = train_dataset,
      test_datset = test_dataset,
      **cfg)"""

model_path = './models/model.pth'
model.load_state_dict(torch.load(model_path, weights_only=True))

#plot_dataset(full_dataset)


_ = plot_probs(model, 100, [0,0], 2, 'energies', analytical = True, params = params)


Z = plot_probs(model, 1000, center = [0,0], hw = 1, values_type = 'probs', analytical = False)

kl = compute_KL(x, model, params, Z)
print(kl)
