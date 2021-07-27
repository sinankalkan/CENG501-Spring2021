import torch
import torch.nn as nn


class seq2seq(nn.Module):
    def __init__(self, hidden_size, output_size, max_length=13805):
        super(seq2seq, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        self.W_h = nn.Parameter(torch.Tensor(self.hidden_size))
        self.W_s = nn.Parameter(torch.Tensor(self.hidden_size))
        self.W_q = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.lstm = nn.LSTM(100, self.hidden_size, bidirectional=False)

    def forward(self, H_a, H_q):
        W_h = self.W_h.repeat(1, H_a.shape[1])
        W_h = W_h.view(self.hidden_size, H_a.shape[1])
        W_s = self.W_s.repeat(1, H_a.shape[1])
        W_s = W_s.view(self.hidden_size, H_a.shape[1])
        W_q = self.W_q.repeat(1, H_a.shape[1])
        W_q = W_q.view(self.hidden_size, H_a.shape[1])
        bias = self.bias.repeat(1, H_a.shape[1])
        bias = bias.view(self.hidden_size, H_a.shape[1])

        s = self.lstm(H_a)
        o_q = torch.mean(H_q)
        e = torch.tanh(W_h @ H_a + W_s @ s + W_q @ o_q + bias)
        a = torch.softmax(e)

        H_t = torch.mul(torch.transpose(a), H_a)
        torch.sum(H_t, dim=0)

        return a
