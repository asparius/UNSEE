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


# loss fn
def loss_fn(x, y):
	x = F.normalize(x, dim=-1, p=2)
	y = F.normalize(y, dim=-1, p=2)
	return 2 - 2 * (x * y).sum(dim=-1)
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()




class VICRegLoss(nn.Module):
	def __init__(
				self,
				model: SentenceTransformer,
				sentence_embedding_dimension: int,
				hidden_embedding_dim: int,
				num_mlp_layers: int = 4,
				mse_weight: float = 25. ,
				var_weight: float = 25. ,
				cov_weight: float = 1.  ,
				moving_average_decay : float= 0.999,
				) -> None:
			super(VICRegLoss, self).__init__()
			self.online_encoder = model
			layers = []
			for _ in range(num_mlp_layers):
				layers.append(MLP(sentence_embedding_dimension, sentence_embedding_dimension , hidden_embedding_dim))

				self.MLPS = nn.Sequential(*layers)
			
			self.target_encoder = copy.deepcopy(self.online_encoder)

			self.mse_weight = mse_weight
			self.var_weight = var_weight
			self.cov_weight = cov_weight

			self.target_ema_updater = EMA(moving_average_decay)  



	def vicreg_loss(self,x,y):
		
		batch_dim, hid_dim = x.shape
		repr_loss = F.mse_loss(x, y)


		x = x - x.mean(dim=0)
		y = y - y.mean(dim=0)

		std_x = torch.sqrt(x.var(dim=0) + 0.0001)
		std_y = torch.sqrt(y.var(dim=0) + 0.0001)
		std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

		cov_x = (x.T @ x) / (batch_dim - 1)
		cov_y = (y.T @ y) / (batch_dim - 1)
		cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
			hid_dim
		) + off_diagonal(cov_y).pow_(2).sum().div(hid_dim)

		loss = (
			self.mse_weight * repr_loss
			+ self.var_weight * std_loss
			+  self.cov_weight * cov_loss
		)
		return loss


	def update_moving_average(self):
		assert self.target_encoder is not None, 'target encoder has not been created yet'
		update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
		
		

	def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):

		target_sentence_features = copy.deepcopy(sentence_features)
		rep_one, rep_two = [self.online_encoder(sentence_feature) for sentence_feature in sentence_features]
		online_pred_one, online_pred_two = rep_one['sentence_embedding'], rep_two['sentence_embedding']
		online_pred_one, online_pred_two = self.MLPS(online_pred_one), self.MLPS(online_pred_two)
		
		with torch.no_grad():

			target_one, target_two = [self.target_encoder(sentence_feature) for sentence_feature in target_sentence_features]
			target_proj_one, target_proj_two = target_one['sentence_embedding'],  target_two['sentence_embedding']
		target_pred_one, target_pred_two = target_proj_one.detach(), target_proj_two.detach()
		
		
		loss_one = self.vicreg_loss(online_pred_one, target_pred_two)
		loss_two = self.vicreg_loss(online_pred_two, target_pred_one)
		

		loss = loss_one + loss_two
		
		return loss.mean()

