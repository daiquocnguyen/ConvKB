import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from Model import Model
from numpy.random import RandomState

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

class ConvKB(Model):

    def __init__(self, config):
        super(ConvKB, self).__init__(config)

        self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size) 
        self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)

        self.conv1_bn = nn.BatchNorm2d(1)
        self.conv_layer = nn.Conv2d(1, self.config.out_channels, (self.config.kernel_size, 3))  # kernel size x 3
        self.conv2_bn = nn.BatchNorm2d(self.config.out_channels)
        self.dropout = nn.Dropout(self.config.convkb_drop_prob)
        self.non_linearity = nn.ReLU() # you should also tune with torch.tanh() or torch.nn.Tanh()
        self.fc_layer = nn.Linear((self.config.hidden_size - self.config.kernel_size + 1) * self.config.out_channels, 1, bias=False)

        self.criterion = nn.Softplus()
        self.init_parameters()

    def init_parameters(self):
        if self.config.use_init_embeddings == False:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

        else:
            self.ent_embeddings.weight.data = self.config.init_ent_embs
            self.rel_embeddings.weight.data = self.config.init_rel_embs

        nn.init.xavier_uniform_(self.fc_layer.weight.data)
        nn.init.xavier_uniform_(self.conv_layer.weight.data)

    def _calc(self, h, r, t):
        h = h.unsqueeze(1) # bs x 1 x dim
        r = r.unsqueeze(1)
        t = t.unsqueeze(1)

        conv_input = torch.cat([h, r, t], 1)  # bs x 3 x dim
        conv_input = conv_input.transpose(1, 2)
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)
        conv_input = self.conv1_bn(conv_input)
        out_conv = self.conv_layer(conv_input)
        out_conv = self.conv2_bn(out_conv)
        out_conv = self.non_linearity(out_conv)
        out_conv = out_conv.view(-1, (self.config.hidden_size - self.config.kernel_size + 1) * self.config.out_channels)
        input_fc = self.dropout(out_conv)
        score = self.fc_layer(input_fc).view(-1)

        return -score

    def loss(self, score, regul):
        return torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul

    def forward(self):
        h = self.ent_embeddings(self.batch_h)
        r = self.rel_embeddings(self.batch_r)
        t = self.ent_embeddings(self.batch_t)
        score = self._calc(h, r, t)

        # regularization
        l2_reg = torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)
        for W in self.conv_layer.parameters():
            l2_reg = l2_reg + W.norm(2)
        for W in self.fc_layer.parameters():
            l2_reg = l2_reg + W.norm(2)

        return self.loss(score, l2_reg)

    def predict(self):

        h = self.ent_embeddings(self.batch_h)
        r = self.rel_embeddings(self.batch_r)
        t = self.ent_embeddings(self.batch_t)
        score = self._calc(h, r, t)

        return score.cpu().data.numpy()
