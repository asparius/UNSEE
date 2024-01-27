
import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import torch.nn.functional as F
from ..SentenceTransformer import SentenceTransformer
import torch
import torch.nn as nn
import numpy as np
import logging
import math
from functools import wraps
import copy
import random


def invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Attraction factor of CorInfoMax Loss: MSE loss calculation from outputs of the projection network, z1 (NXD) from 
    the first branch and z2 (NXD) from the second branch. Returns loss part comes from attraction factor (mean squared error).
    """
    return F.mse_loss(z1, z2)





class CovarianceLoss(nn.Module):
    """Big-bang factor of CorInfoMax Loss: loss calculation from outputs of the projection network,
    z1 (NXD) from the first branch and z2 (NXD) from the second branch. Returns loss part comes from bing-bang factor.
    """

    def __init__(self, proj_output_dim,R_ini,la_R,la_mu,R_eps_weight):
        super(CovarianceLoss, self).__init__()
        
        proj_output_dim = proj_output_dim
        self.R1 = R_ini * torch.eye(
            proj_output_dim, dtype=torch.float64, device="cuda", requires_grad=False
        )
        self.mu1 = torch.zeros(
            proj_output_dim, dtype=torch.float64, device="cuda", requires_grad=False
        )
        self.R2 = R_ini * torch.eye(
            proj_output_dim, dtype=torch.float64, device="cuda", requires_grad=False
        )
        self.mu2 = torch.zeros(
            proj_output_dim, dtype=torch.float64, device="cuda", requires_grad=False
        )
        self.new_R1 = torch.zeros(
            (proj_output_dim, proj_output_dim),
            dtype=torch.float64,
            device="cuda",
            requires_grad=False,
        )
        self.new_mu1 = torch.zeros(
            proj_output_dim, dtype=torch.float64, device="cuda", requires_grad=False
        )
        self.new_R2 = torch.zeros(
            (proj_output_dim, proj_output_dim),
            dtype=torch.float64,
            device="cuda",
            requires_grad=False,
        )
        self.new_mu2 = torch.zeros(
            proj_output_dim, dtype=torch.float64, device="cuda", requires_grad=False
        )
        self.la_R = la_R
        self.la_mu = la_mu
        self.R_eigs = torch.linalg.eigvals(self.R1).unsqueeze(0)
        self.R_eps_weight = R_eps_weight
        self.R_eps = self.R_eps_weight * torch.eye(
            proj_output_dim, dtype=torch.float64, device="cuda", requires_grad=False
        )

        
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        la_R = self.la_R
        la_mu = self.la_mu

        N, D = z1.size()

        # mean estimation
        mu_update1 = torch.mean(z1, 0)
        mu_update2 = torch.mean(z2, 0)
        self.new_mu1 = la_mu * (self.mu1) + (1 - la_mu) * (mu_update1)
        self.new_mu2 = la_mu * (self.mu2) + (1 - la_mu) * (mu_update2)

        # covariance matrix estimation
        z1_hat = z1 - self.new_mu1
        z2_hat = z2 - self.new_mu2
        R1_update = (z1_hat.T @ z1_hat) / N
        R2_update = (z2_hat.T @ z2_hat) / N
        self.new_R1 = la_R * (self.R1) + (1 - la_R) * (R1_update)
        self.new_R2 = la_R * (self.R2) + (1 - la_R) * (R2_update)

        # loss calculation
        cov_loss = (
            -(
                torch.logdet(self.new_R1 + self.R_eps)
                + torch.logdet(self.new_R2 + self.R_eps)
            )
            / D
        )

        # This is required because new_R updated with backward.
        self.R1 = self.new_R1.detach()
        self.mu1 = self.new_mu1.detach()
        self.R2 = self.new_R2.detach()
        self.mu2 = self.new_mu2.detach()

        return cov_loss


class EMA():
	def __init__(self, beta):
		super().__init__()
		self.beta = beta

	def update_average(self, old, new):
		if old is None:
			return new
		return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
	for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
		old_weight, up_weight = ma_params.data, current_params.data
		ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP for  predictor
class MLP(nn.Module):
	def __init__(self, dim, projection_size, hidden_size):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(dim, hidden_size),
			nn.BatchNorm1d(hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, projection_size)
		)

	def forward(self, x):
		return self.net(x)




class CorInfoMaxLoss(nn.Module):
	def __init__(self,
				 model: SentenceTransformer,
				 sentence_embedding_dimension: int,
				 hidden_embedding_dim: int ,
				 num_mlp_layers: int = 4,
				 R_ini: float = 1,
				 la_R: float = 0.01,
				 la_mu: float = 0.01,
				 R_eps_weight: float = 1e-6,
				 cov_weight = 0.2,
				 inv_weight = 2000,
				 moving_average_decay: float = 0.999):
		super(CorInfoMaxLoss, self).__init__()
		self.online_encoder = model
		layers = []
		for _ in range(num_mlp_layers):
			layers.append(MLP(sentence_embedding_dimension, sentence_embedding_dimension , hidden_embedding_dim))

		self.MLPS = nn.Sequential(*layers)

		self.target_encoder = copy.deepcopy(self.online_encoder)
	
		self.cov_loss_fct = CovarianceLoss(proj_output_dim=sentence_embedding_dimension,R_ini=R_ini,la_R=la_R,la_mu=la_mu,R_eps_weight=R_eps_weight)
		self.inv_loss_fct = invariance_loss
		self.cov_weight = cov_weight
		self.inv_weight = inv_weight
		self.target_ema_updater = EMA(moving_average_decay)  
		#self.batch_norm = nn.BatchNorm1d(sentence_embedding_dimension)
	def update_moving_average(self):
		assert self.target_encoder is not None, 'target encoder has not been created yet'
		update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)


	def corinfomaxloss(self,z1,z2):

		cov_loss = self.cov_loss_fct(z1, z2)
		sim_loss = self.inv_loss_fct(z1, z2)
		loss = cov_loss * self.cov_weight  + self.inv_weight * sim_loss

		return loss

	def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):

		target_sentence_features = copy.deepcopy(sentence_features)
		rep_one, rep_two = [self.online_encoder(sentence_feature) for sentence_feature in sentence_features]
		online_pred_one, online_pred_two = rep_one['sentence_embedding'], rep_two['sentence_embedding']
		online_pred_one, online_pred_two = self.MLPS(online_pred_one), self.MLPS(online_pred_two)

		
		with torch.no_grad():

			target_one, target_two = [self.target_encoder(sentence_feature) for sentence_feature in target_sentence_features]
			target_proj_one, target_proj_two = target_one['sentence_embedding'],  target_two['sentence_embedding']
	
		
		target_pred_one,target_pred_two = target_proj_one.detach(), target_proj_two.detach()

		loss_one = self.corinfomaxloss(online_pred_one, target_pred_two)
		loss_two = self.corinfomaxloss(online_pred_two,target_pred_one)
		
		loss = loss_one + loss_two			

		return loss.mean()

