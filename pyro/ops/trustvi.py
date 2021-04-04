import torch
from torch.autograd import grad
from torch.autograd.functional import hessian 




def propose_step(loss, x):
    """
    Proposes a step to be evaluated by {replace here}. 

    :param torch.Tensor loss: A scalar function ``x`` to be minimized. 
    :param torch.Tensor x: A dependent variable with rightmost size of 2. 
    
    :list proposed_step: List of proposed steps for each parameter in x

    Note: 
    For the moment I assume that every parameter is a vector (in the mathematical sense)
    """

    if loss.shape != ():
        raise ValueError('Expected loss to be a scalar, actual shape{}'.format(loss.shape)) 

    x = list(x)
    proposed_step = []
    g = grad(loss, x, create_graph=True)

    # We need to compute the hessians idependently in case the have different shapes
    # For the moment I am assuming that every parameter is a row
    # We can probably handle this later

    for i, param in enumerate(x):
        hessian_rows = [grad(g[i][j], param, retain_graph=True)[0] for j in range(len(param))]
        H = torch.stack(hessian_rows)
        H_inverse = torch.inverse(H)
        proposed_step.append(- H_inverse @ g[i])
        
    return proposed_step
                 
