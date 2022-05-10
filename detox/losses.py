import torch


def cross_entropy(input, target):
    return -torch.sum(target * torch.log(input) + (1. - target) * torch.log(1. - input))