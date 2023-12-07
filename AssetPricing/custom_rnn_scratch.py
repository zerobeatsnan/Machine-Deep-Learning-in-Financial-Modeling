import torch
import math
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import collections
import re
import random




def get_custom_rnn_params(input_size, inner_structure, hidden_size, output_size, device):
    
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    all_sizes = [input_size + hidden_size] + inner_structure + [hidden_size]
    params = []

    for i in range(len(all_sizes) - 1):
        W = normal((all_sizes[i], all_sizes[i+1]))
        b = torch.zeros(all_sizes[i+1], device=device)
        params.extend([W, b])

    W_hq = normal((hidden_size, output_size))

    b_q = torch.zeros(output_size, device = device)

    params.extend([W_hq, b_q])

    for param in params:
        param.requires_grad_(True)

    return params

def custom_rnn_forward(inputs, state, params, inner_structure):
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

class CustomRNNFromScratch:
    def __init__(self, input_size, inner_structure, hidden_size, output_size, device):
        self.params = get_custom_rnn_params(input_size, inner_structure, hidden_size, output_size, device)
        self.inner_structure = inner_structure
        self.hidden_size = hidden_size
        self.input_size = input_size

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.input_size).type(torch.float32)
        return custom_rnn_forward(X, state, self.params, self.inner_structure)

    def begin_state(self, batch_size, device):
        return (torch.zeros((batch_size, self.hidden_size), device=device),)
