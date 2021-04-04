import torch
import trustvi as tvi
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import numpy as np

from pyro.poutine.runtime import effectful
from pyro.infer.autoguide import AutoDiagonalNormal


def model(obs_values=None):
    weight = pyro.sample("weight", dist.Normal(torch.ones(2), torch.tensor(1.0)))
    weight = torch.ones((100,2)) * weight
    variance = pyro.sample('variance', dist.Normal(torch.ones(1), torch.tensor(10.0)))
    values = pyro.sample('measurement', dist.Normal(weight, variance**2), obs = obs_values)

    return values, weight, variance

def guide(obs_values=None):
    mean_weight = pyro.param('mean_weight', torch.ones(2))
    mean_std = pyro.param('mean_std', torch.ones(1))                           
    pyro.sample('weight', dist.Normal(mean_weight, torch.tensor(1.0)))
    pyro.sample('variance', dist.Normal(mean_std, torch.tensor(1.0)))


elbo = pyro.infer.Trace_ELBO()



def train(model, guide, data):
    with poutine.trace(param_only=True) as param_capture:
        loss = elbo.differentiable_loss(model, guide,data)

    params = set(site["value"].unconstrained()
                 for site in param_capture.trace.nodes.values())


    tvi.propose_step(loss, params)
    print(params)

    



data, weight, variance = model()

print('Test 1')
train(model, guide, data)


x = torch.tensor(np.array([1.0,2]), requires_grad=True)
y = torch.tensor(np.array([4.0]), requires_grad=True)



print('Test 2')
result = torch.sum(x**2) + y**3
H = tvi.propose_step(result[0], [x,y])
print(H)


# adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
# optimizer = pyro.optim.Adam(adam_params)

# # setup the inference algorithm
# svi = pyro.infer.SVI(model, guide, optimizer, loss=elbo)

# n_steps = 5000
# # do gradient steps
# for step in range(n_steps):
#     svi.step(data)

# print(weight[0])
