import torch
import copy
from torch import nn, Tensor
from typing import Iterable, Dict
from ..SentenceTransformer import SentenceTransformer
from .. import util
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

class MultipleNegativesRankingLoss(nn.Module):
    """
        This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_n, p_n)
        where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.

        For each a_i, it uses all other p_j as negative samples, i.e., for a_i, we have 1 positive example (p_i) and
        n-1 negative examples (p_j). It then minimizes the negative log-likehood for softmax normalized scores.

        This loss function works great to train embeddings for retrieval setups where you have positive pairs (e.g. (query, relevant_doc))
        as it will sample in each batch n-1 negative docs randomly.

        The performance usually increases with increasing batch sizes.

        For more information, see: https://arxiv.org/pdf/1705.00652.pdf
        (Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4)

        You can also provide one or multiple hard negatives per anchor-positive pair by structering the data like this:
        (a_1, p_1, n_1), (a_2, p_2, n_2)

        Here, n_1 is a hard negative for (a_1, p_1). The loss will use for the pair (a_i, p_i) all p_j (j!=i) and all n_j as negatives.

        Example::

            from sentence_transformers import SentenceTransformer, losses, InputExample
            from torch.utils.data import DataLoader

            model = SentenceTransformer('distilbert-base-uncased')
            train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']),
                InputExample(texts=['Anchor 2', 'Positive 2'])]
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
            train_loss = losses.MultipleNegativesRankingLoss(model=model)
    """
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct = util.cos_sim,moving_average_decay=0.99,sentence_embedding_dimension= 768):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(MultipleNegativesRankingLoss, self).__init__()
        self.online_encoder = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_ema_updater = EMA(moving_average_decay)
        self.online_predictor_1 = MLP(sentence_embedding_dimension, sentence_embedding_dimension , sentence_embedding_dimension * 10) 
        self.online_predictor_2 = MLP(sentence_embedding_dimension ,sentence_embedding_dimension ,sentence_embedding_dimension * 10) 
        self.online_predictor_3 = MLP(sentence_embedding_dimension , sentence_embedding_dimension, sentence_embedding_dimension * 10) 
        self.online_predictor_4 = MLP(sentence_embedding_dimension , sentence_embedding_dimension, sentence_embedding_dimension * 10)
        
    
    
    def update_moving_average(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        
        
        rep_one, rep_two = [self.online_encoder(sentence_feature) for sentence_feature in sentence_features]
        online_pred_one, online_pred_two = rep_one['sentence_embedding'], rep_two['sentence_embedding']
        online_pred_one, online_pred_two = self.online_predictor_1(online_pred_one), self.online_predictor_1(online_pred_two)
        online_pred_one, online_pred_two = self.online_predictor_2(online_pred_one), self.online_predictor_2(online_pred_two)
        online_pred_one, online_pred_two = self.online_predictor_3(online_pred_one), self.online_predictor_3(online_pred_two)
        online_pred_one, online_pred_two = self.online_predictor_4(online_pred_one), self.online_predictor_4(online_pred_two)
        #online_pred_one, online_pred_two = self.online_predictor_5(online_pred_one), self.online_predictor_5(online_pred_two)
        """
        with torch.no_grad():

            target_one, target_two = [self.target_encoder(sentence_feature) for sentence_feature in sentence_features]
            target_proj_one, target_proj_two = target_one['sentence_embedding'],  target_two['sentence_embedding']
        
        target_pred_one, target_pred_two = target_proj_one.detach(), target_proj_two.detach()
        #target_pred_one, target_pred_two = self.online_predictor_1(target_pred_one), self.online_predictor_1(target_pred_two)
        #target_pred_one, target_pred_two = self.online_predictor_2(target_pred_one), self.online_predictor_2(target_pred_two)
        #target_pred_one, target_pred_two = self.online_predictor_3(target_pred_one), self.online_predictor_3(target_pred_two)
        #target_pred_one, target_pred_two = self.online_predictor_4(target_pred_one), self.online_predictor_4(target_pred_two)
        """
        scores = self.similarity_fct(online_pred_one, online_pred_two) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
        loss = self.cross_entropy_loss(scores, labels)
        """
        scores1 = self.similarity_fct(online_pred_one,target_pred_two) * self.scale
        scores2 = self.similarity_fct(online_pred_two,target_pred_one) * self.scale
        labels = torch.tensor(range(len(scores1)), dtype=torch.long, device=scores1.device)
        loss1 = self.cross_entropy_loss(scores1,labels)
        loss2= self.cross_entropy_loss(scores2,labels)
        loss = (loss1 + loss2) / 2
        """
        return loss
    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__}





