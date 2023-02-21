"""
Models module for SVP experiments

Author: Thomas Mortier
Date: January 2022
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from itertools import product

class Identity(nn.Module):
    """ Identitify layer.
    """
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def get_phi_text(args):
    phi = torch.nn.Sequential(
            torch.nn.Linear(args.dim, args.hidden),
            torch.nn.ReLU()
            )

    return phi

def get_phi_caltech(args):
    phi = GET_MODEL[args.phi](pretrained=True)
    phi.fx = Identity()

    return phi

def get_phi_plantclef(args):
    phi = GET_MODEL[args.phi](pretrained=True)
    phi.fx = Identity()

    return phi

""" dictionary representing model mapper """
GET_MODEL = {
    "MOB": models.mobilenet_v2,
    "VGG": models.vgg16
}

""" dictionary representing dataset->phi mapper """
GET_PHI = {
    "CIFAR10": get_phi_caltech,
    "CAL101": get_phi_caltech,
    "CAL256": get_phi_caltech,
    "PLANTCLEF": get_phi_plantclef,
    "PROT": get_phi_text,
    "BACT": get_phi_text,
}

""" calculate accuracy given predictions and labels """
def accuracy(predictions, labels):
    if len(predictions) > 1:
        o = (np.array(predictions)==np.array(labels))
    else:
        o = (np.array(predictions)==np.array(labels))

    return np.mean(o)
