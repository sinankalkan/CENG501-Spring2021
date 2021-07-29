import torch
import torch.nn as nn
from math import sqrt


class PreProcess(nn.Module):

    def __init__(self, dimentionality, sequence_length):
        super().__init__()
        self.dim = dimentionality
        self.seq_len = sequence_length

        self.W_i = nn.Parameter(torch.Tensor(self.dim, self.seq_len))
        self.W_u = nn.Parameter(torch.Tensor(self.dim, self.seq_len))
        self.W_g = nn.Parameter(torch.Tensor(self.seq_len, self.seq_len))
        self.b_i = nn.Parameter(torch.Tensor(self.seq_len))
        self.b_u = nn.Parameter(torch.Tensor(self.seq_len))
        self.b_g = nn.Parameter(torch.Tensor(self.seq_len))


        self.initialize_weights()

    def initialize_weights(self):
        stdv = 1.0 / sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, A, Q):
        """

        """
        dim, ans_sequence = A.size()
        dim, ques_sequence = Q.size()

        bias_ia = self.b_i.repeat(1, ans_sequence)
        bias_ia = bias_ia.view(self.dim, ans_sequence)
        bias_ua = self.b_u.repeat(1, ans_sequence)
        bias_ua = bias_ua.view(self.dim, ans_sequence)
        bias_iq = self.b_i.repeat(1, ques_sequence)
        bias_iq = bias_iq.view(self.dim, ques_sequence)
        bias_uq = self.b_u.repeat(1, ques_sequence)
        bias_uq = bias_uq.view(self.dim, ques_sequence)
        a_i = torch.sigmoid(self.W_i @ A + bias_ia)
        a_u = torch.tanh(self.W_u @ A + bias_ua)
        q_i = torch.sigmoid(self.W_i @ Q + bias_iq)
        q_u = torch.sigmoid(self.W_i @ Q + bias_uq)

        _A = torch.mul(a_i, a_u)
        _Q = torch.mul(q_i, q_u)
        bias_g = self.b_g.repeat(1, ques_sequence)
        bias_g = bias_g.view(self.dim, ques_sequence)
        G = torch.softmax(torch.mul(torch.transpose(self.W_g @ _Q + bias_g, 0, 1), _A), dim=1)
        H = torch.mul(_Q, G)

        return H, G

