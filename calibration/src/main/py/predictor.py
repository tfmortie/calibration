"""
Implementation for PyTorch predictor.

Author: Thomas Mortier
Date: March 2022
"""
import torch
import cvxpy
import time
import torch.nn as nn
import numpy as np

from nnet_cpp import NNet

class Predictor(torch.nn.Module):
    """ Pytorch module which represents a predictor.

    Parameters
    ----------
    phi : torch.nn.Module
        Represents the neural network architecture which learns the hidden representations
        for the probabilistic model. Must be of type torch.nn.Module with output 
        (batch_size, hidden_size).
    hidden_size : int
        Size of the hidden representation which is passed to the probabilistic model.
    dropout : float
        Dropout probability for last layer
    num_classes : int 
        Number of classes.
    hstruct : nested list of int or tensor, default=None
        Hierarchical structure of the classification problem. If None,
        a flat probabilistic model is going to be considered.
    transformer : Transformer, default=None
        Transformer needed for the calibration module. Is None in case of a flat
        probabilistic model.

    Attributes
    ----------
    phi : torch.nn.Module
        Represents the neural network architecture which learns the hidden representation
        for the probabilistic model. Must be of type torch.nn.Module with output 
        (batch_size, hidden_size).
    hidden_size : int
        Size of the hidden representation which is passed to the probabilistic model.
    dropout : float
        Dropout probability for last layer
    num_classes : int 
        Number of classes.
    hstruct : nested list of int or tensor, default=None
        Hierarchical structure of the classification problem. If None,
        a flat probabilistic model is going to be considered.
    transformer : Transformer, default=None
        Transformer needed for the calibration module. Is None in case of a flat
        probabilistic model.
    nnet : NNet module
        NNet module.
    """
    def __init__(self, phi, hidden_size, dropout, num_classes, hstruct=None, transformer=None):
        super(Predictor, self).__init__()
        self.phi = phi
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_classes = num_classes
        self.hstruct = hstruct
        if self.hstruct is None:
            h = []
        else:
            h = self.hstruct
        self.nnet = NNet(self.hidden_size, self.num_classes, self.dropout, h)
        self.transformer = transformer

    def forward(self, x, y=None):
        """ Forward pass for the predictor.
        
        Parameters
        ----------
        x : input tensor of size (N, D) 
            Represents a batch.
        y : target tensor or list, default=None
            Represents the target labels.

        Returns
        -------
        o : loss tensor of size (N,)
        """
        # get embeddings
        x = self.phi(x)
        if y is not None:
            if type(y) is torch.Tensor:
                y = y.tolist()
            # inverse transform labels
            if type(self.hstruct) is list and len(self.hstruct)>0:
                y = self.transformer.transform(y, True)
            else:
                y = self.transformer.transform(y, False)
                y = torch.Tensor(sum(y,[])).long().to(x.device)
            o = self.nnet(x,y)
        else:
            o = self.nnet(x)

        return o
    
    def predict_proba(self, x):
        """ Predict_proba function for the predictor.
        
        Parameters
        ----------
        x : input tensor of size (N, D) 
            Represents a batch.

        Returns
        -------
        p :  output tensor representing probabilities
        """
        x = self.phi(x)
        p = self.nnet.predict_proba(x)

        return p
    
    def predict(self, x):
        """ Predict function for the predictor.
        
        Parameters
        ----------
        x : input tensor of size (N, D) 
            Represents a batch.

        Returns
        -------
        o : list of outputs of size (N,)
        """
        x = self.phi(x)
        o = self.nnet.predict(x)
        o = self.transformer.inverse_transform(o)

        return o
