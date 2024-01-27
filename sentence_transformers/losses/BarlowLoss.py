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



def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()




class BarlowLoss(nn.Module):
	def __init__(self,
				 model: SentenceTransformer,
				 sentence_embedding_dimension: int,
				 hidden_embedding_dim: int,
				 num_mlp_layers: int = 4,
				 off_weight: float = 0.0051,
				 moving_average_decay: float = 0.999):
		super(BarlowLoss, self).__init__()
		self.online_encoder = model

		layers = []
		for _ in range(num_mlp_layers):
			layers.append(MLP(sentence_embedding_dimension, sentence_embedding_dimension , hidden_embedding_dim))

		self.MLPS = nn.Sequential(*layers)
		self.target_encoder = copy.deepcopy(self.online_encoder)
		self.off_weight = off_weight
		

		self.target_ema_updater = EMA(moving_average_decay)  
		self.batch_norm = nn.BatchNorm1d(sentence_embedding_dimension)
	def update_moving_average(self):
		assert self.target_encoder is not None, 'target encoder has not been created yet'
		update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

	def barlow_loss(self,x,y):
		x,y = self.batch_norm(x),self.batch_norm(y)
		batch_size,_ = x.shape
		c = x.T @ y
		c.div_(batch_size)

		on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
		off_diag = off_diagonal(c).pow_(2).sum()
		loss = on_diag + self.off_weight * off_diag
		return loss

		

	def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):

		target_sentence_features = copy.deepcopy(sentence_features)
		rep_one, rep_two = [self.online_encoder(sentence_feature) for sentence_feature in sentence_features]
		online_pred_one, online_pred_two = rep_one['sentence_embedding'], rep_two['sentence_embedding']
		online_pred_one, online_pred_two = self.MLPS(online_pred_one), self.MLPS(online_pred_two)
		
		with torch.no_grad():

			target_one, target_two = [self.target_encoder(sentence_feature) for sentence_feature in target_sentence_features]
			target_proj_one, target_proj_two = target_one['sentence_embedding'],  target_two['sentence_embedding']
		target_pred_one, target_pred_two = target_proj_one.detach(), target_proj_two.detach()

		
		loss_one = self.barlow_loss(online_pred_one, target_pred_two)
		loss_two = self.barlow_loss(online_pred_two, target_pred_one)


		loss = loss_one + loss_two
		

		return loss.mean()

