import torch
import torch.nn.functional as F
from logging import log
import logging
from utils_models import SparsyFed_no_act_Conv1D, SparsyFed_no_act_Conv2D, SparsyFed_no_act_linear, SparsyFedConv2D, SparsyFedLinear, SWATConv2D,SWATLinear
import torch.nn as nn
import numpy as np
import math
from scipy import special

class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = self.kl_div(p_s, p_t) * (self.T ** 2)
        return loss
    
def sigmoid_weight(r,R,k):
    return 1 / (1 + np.exp(-k*(r-R/2)))


def disco_weight_adjusting(old_weight, distribution_difference, a, b):
    weight_temp = old_weight - a * distribution_difference + b
    if np.sum(weight_temp > 0) > 0:
        new_weight = np.copy(weight_temp)
        new_weight[weight_temp < 0] = 0
    else:
        new_weight = np.copy(old_weight)
    total_normalizer = sum([new_weight[r] for r in range(len(old_weight))])
    new_weight = [new_weight[r] / total_normalizer for r in range(len(old_weight))]
    return new_weight


def get_distribution_difference(client_cls_counts, selected_clients, metric, hypo_distribution):
    local_distributions = client_cls_counts[np.array(selected_clients)]
    local_distributions = local_distributions / np.sum(local_distributions, axis=1)[:,np.newaxis]
    if metric == 'consine':
        similarity_scores = local_distributions.dot(hypo_distribution) / (np.linalg.norm(local_distributions, axis=1) * np.linalg.norm(hypo_distribution))
        difference = 1.0 - similarity_scores
    elif metric == 'only_iid':
        similarity_scores = local_distributions.dot(hypo_distribution) / (np.linalg.norm(local_distributions, axis=1) * np.linalg.norm(hypo_distribution))
        difference = np.where(similarity_scores > 0.9, 0.01, float('inf'))
    elif metric == 'l1':
        difference = np.linalg.norm(local_distributions - hypo_distribution, ord=1, axis=1)
    elif metric == 'l2':
        difference = np.linalg.norm(local_distributions - hypo_distribution, axis=1)
    elif metric == 'kl':
        difference = special.kl_div(local_distributions, hypo_distribution)
        difference = np.sum(difference, axis=1)
        difference = np.array([0 for _ in range(len(difference))]) if np.sum(difference) == 0 else difference / np.sum(difference)
    return difference

def get_mdl_params(model):
    # model parameters ---> vector (different storage)
    vec = []
    for param in model.parameters():
        vec.append(param.clone().detach().cpu().reshape(-1))
    return torch.cat(vec)

def set_clnt_from_params(device, model, params):
    idx = 0
    for param in model.parameters():
        length = param.numel()
        param.data.copy_(params[idx:idx + length].reshape(param.shape))
        idx += length
    return model.to(device)

def param_to_vector(model):
    # model parameters ---> vector (same storage)
    vec = []
    for param in model.parameters():
        vec.append(param.reshape(-1))
    return torch.cat(vec)
    


def set_client_from_params(device, model, params):
    idx = 0
    for param in model.parameters():
        length = param.numel()
        param.data.copy_(params[idx:idx + length].reshape(param.shape))
        idx += length
    return model.to(device)



def get_params_list_with_shape(model, param_list):
    vec_with_shape = []
    idx = 0
    for param in model.parameters():
        length = param.numel()
        vec_with_shape.append(param_list[idx:idx + length].reshape(param.shape))
    return vec_with_shape


def get_params_to_prune(model, first_layer=False):
    """
    Get parameters to prune in the model.
    """
    params_to_prune = []
    first_layer = first_layer
    def add_immediate_child(
        module: nn.Module,
        name: str,
    ) -> None:
        nonlocal first_layer
        if (
            type(module) == SparsyFed_no_act_Conv2D
            or type(module) == SparsyFed_no_act_Conv1D
            or type(module) == SparsyFed_no_act_linear
            or type(module) == SparsyFedConv2D
            or type(module) == SparsyFedLinear
            or type(module) == SWATConv2D
            or type(module) == SWATLinear
            or type(module) == nn.Conv2d
            or type(module) == nn.Conv1d
            or type(module) == nn.Linear
        ):
            if first_layer:
                first_layer = False
            else:
                params_to_prune.append((module, "weight", name))
        for _name, immediate_child_module in module.named_children():
            add_immediate_child(immediate_child_module, _name)
    add_immediate_child(model, "Net")

    return params_to_prune