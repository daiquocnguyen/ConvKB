import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Model(nn.Module):
	def __init__(self, config):
		super(Model, self).__init__()
		self.config = config
		self.batch_h = None
		self.batch_t = None
		self.batch_r = None
		self.batch_y = None

	def get_positive_score(self, score):
		return score[0:self.config.batch_size]

	def get_negative_score(self, score):
		negative_score = score[self.config.batch_size:self.config.batch_seq_size]
		negative_score = negative_score.view(-1, self.config.batch_size)
		negative_score = torch.mean(negative_score, 0)
		return negative_score
	
	def forward(self):
		raise NotImplementedError
	
	def predict(self):
		raise NotImplementedError	
