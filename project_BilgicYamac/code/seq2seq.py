import torch
import torch.nn as nn
from math import sqrt


class seq2seq(nn.Module):
    def __init__(self, width):
        super(seq2seq, self).__init__()
        self.width = width

        self.W_h = nn.Parameter(torch.Tensor(self.width))
        self.W_s = nn.Parameter(torch.Tensor(150))
        self.W_q = nn.Parameter(torch.Tensor(self.width))
        self.bias = nn.Parameter(torch.Tensor(self.width))
        self.lstm = nn.LSTM(300, 150, bidirectional=False)

        self.initialize_weights()

    def initialize_weights(self):
        stdv = 1.0 / sqrt(self.width)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, H_a, H_q):
        W_h = self.W_h.repeat(1, H_a.shape[1])
        W_h = W_h.view(self.width, H_a.shape[1])
        W_s = self.W_s.repeat(1, H_a.shape[1])
        W_s = W_s.view(150, H_a.shape[1])
        W_q = self.W_q.repeat(1, H_a.shape[1])
        W_q = W_q.view(self.width, H_a.shape[1])
        bias = self.bias.repeat(1, H_a.shape[1])
        bias = bias.view(self.width, H_a.shape[1])

        s = self.lstm(H_a)
        o_q = torch.mean(H_q)
        answer = W_h @ H_a[0]
        s_hidden = W_s @ s[0][0]
        question = W_q * o_q
        e = torch.tanh(answer + s_hidden + question + bias)
        a = torch.softmax(e)

        H_t = torch.mul(torch.transpose(a), H_a)
        torch.sum(H_t, dim=0)

        return a


# if __name__ == '__main__':
#     PATH_h_a = "data/sequence/h_a.t7"
#     PATH_h_q= "data/sequence/h_q.t7"
#     h_a = torch.load(PATH_h_a)
#     h_q = torch.load(PATH_h_q)
#     model = seq2seq(h_a.shape[2])
#     a = model(h_a, h_q)