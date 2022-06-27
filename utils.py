import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def grad_vec(pyModel): 
    """
    It returns the gradient of pytorch model as torch tensor vectorized
    """
    gradient_vec = [torch.flatten(w.grad) for w in pyModel.parameters()]
    return torch.cat(gradient_vec )


def weight_vec(pyModel): 
    """
    It returns the weight of pytorch model as torch tensor vectorized
    """
    weight_vec = [torch.flatten(w.data) for w in pyModel.parameters()]
    return torch.cat(weight_vec )
   

def count_parameters(model):
    """
    count the number of parameters for pytorch model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def weight_dec_global(pyModel, weight_vec, eta): 
    """
    Reshape the gradient back to its original shape in pytorch and then 
    performs gradient descent
    """
    c = 0
    for w in pyModel.parameters():
        m = w.numel()
        D = weight_vec[c:m+c].reshape(w.data.shape) 
        c+=m
        if w.data is None:
            w.data = D+0
        else:
            with torch.no_grad():
                w.set_( D+0 )
    return pyModel

def grad_dec_global(pyModel, grad_vec, eta): 
    """
    Reshape the gradient back to its original shape in pytorch and then 
    performs gradient descent
    """
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    c = 0
    for w in pyModel.parameters():
        m = w.numel()
        
        D = grad_vec[c:m+c].reshape(w.data.shape) 
        c+=m
        if w.grad is None:
            w.grad = D+0
        else:
            w.grad.set_( D+0 )