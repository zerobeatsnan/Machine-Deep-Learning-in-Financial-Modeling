import torch
import math
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import collections
import re
import random


def custom_rnn_forward(inputs, state, params):
    outputs = []
    H, = state

    # Here the "input" shape: ("num_steps", "batch_size", "vocab_size")
    for X in inputs:
        combined = torch.cat([X, H], dim=1)
        
        for i in range(0, len(params) - 4, 2):
            W, b = params[i], params[i + 1]
            combined = torch.relu(torch.mm(combined, W) + b)

        W, b = params[-4], params[-3]
        H = torch.tanh(torch.mm(combined, W) + b)

        W_hq, b_q = params[-2], params[-1]
        Y = torch.mm(H, W_hq) + b_q

        outputs.append(Y)

    return torch.cat(outputs, dim=0), (H,)


def custom_rnn_run(X, H, params):

    combined = torch.cat([X, H], dim=1)
    
    for i in range(0, len(params) - 4, 2):
        W, b = params[i], params[i + 1]
        combined = torch.relu(torch.mm(combined, W) + b)

    W, b = params[-4], params[-3]
    H = torch.tanh(torch.mm(combined, W) + b)

    return H


# def get_gru_params(input_size, inner_structure, hidden_size, output_size, device):

#     def normal(shape):

#         return torch.randn(size=shape, device=device) * 0.01

#     all_sizes = [input_size + hidden_size] + inner_structure + [hidden_size]
#     params = []

#     for i in range(len(all_sizes) - 1):
#         W = normal((all_sizes[i], all_sizes[i+1]))
#         b = torch.zeros(all_sizes[i+1], device=device)
#         params.extend([W, b])

#     W_hq = normal((hidden_size, output_size))

#     b_q = torch.zeros(output_size, device = device)

#     params.extend([W_hq, b_q])


#     def three():

#         return (normal((input_size, hidden_size)),
#                 normal((hidden_size, hidden_size)),
#                 torch.zeros(hidden_size, device = device))
    
#     W_xz, W_hz, b_z = three()
#     W_xr, W_hr, b_r = three()

#     params.extend([W_xz, W_hz, b_z, W_xr, W_hr, b_r])

#     for param in params:
#         param.requires_grad_(True)


#     return params



def get_gru_params(input_size, inner_structure, hidden_size, output_size, device):

    def xavier(shape):
        # Create an empty tensor for weights
        weights = torch.empty(shape, device=device)
        # Apply Xavier uniform initialization
        torch.nn.init.xavier_uniform_(weights)
        return weights

    all_sizes = [input_size + hidden_size] + inner_structure + [hidden_size]
    params = []

    for i in range(len(all_sizes) - 1):
        W = xavier((all_sizes[i], all_sizes[i+1]))
        b = torch.zeros(all_sizes[i+1], device=device)
        params.extend([W, b])

    W_hq = xavier((hidden_size, output_size))
    b_q = torch.zeros(output_size, device=device)
    params.extend([W_hq, b_q])

    def three():
        return (xavier((input_size, hidden_size)),
                xavier((hidden_size, hidden_size)),
                torch.zeros(hidden_size, device=device))
    
    W_xz, W_hz, b_z = three()
    W_xr, W_hr, b_r = three()
    params.extend([W_xz, W_hz, b_z, W_xr, W_hr, b_r])

    for param in params:
        param.requires_grad_(True)

    return params




def gru(inputs, state, params):

    W_xz, W_hz, b_z, W_xr, W_hr, b_r = params[-6:]

    rnn_params = params[:-6]

    H, = state

    W_hq, b_q = rnn_params[-2], rnn_params[-1]

    outputs = []

    for X in inputs:

        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = R * H
        H_tilda = custom_rnn_run(X, H_tilda, rnn_params)
        H = Z * H + (1 - Z) * H_tilda
        Y = torch.mm(H, W_hq) + b_q

        outputs.append(Y)

        Y_concat = torch.cat(outputs, dim = 0)

    return Y_concat, (H,)

def grad_clipping(net, theta):  #@save
    """Clip the gradient."""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params

    
    # print([i for i, param in enumerate(params)])
    # print([i for i, param in enumerate(params) if param.grad is None])
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

class CustomGRUromScratch:
    def __init__(self, input_size, inner_structure, hidden_size, output_size, device):
        self.params = get_gru_params(input_size, inner_structure, hidden_size, output_size, device)
        self.inner_structure = inner_structure
        self.hidden_size = hidden_size
        self.input_size = input_size

    def __call__(self, X, state):
        X = X.permute(1,0,2)
        return gru(X, state, self.params)

    def begin_state(self, batch_size, device):
        return (torch.zeros((batch_size, self.hidden_size), device=device),)