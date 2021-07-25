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


def answer_padding(ans, emb, max):
    # map(add_zeros, map(lambda x: x[1], ans))
    new_ans = add_zeros(ans, max)
    return torch.Tensor([torch.Tensor([float(k) for k in emb[int(i)]]) if int(i) != 0 else torch.zeros(100) for i in new_ans])


def add_zeros(ans, max):
    num_zeros = max - len(ans)
    zeros = torch.zeros(num_zeros)
    return torch.cat((ans, zeros), 0)

if __name__ == "__main__":
    PATH_sum = "data/sequence/summary_wikia.t7"
    PATH_train = "data/sequence/train_wikia.t7"
    PATH_emb = "data/sequence/initEmb.t7"
    summ = torch.load(PATH_sum)
    train = torch.load(PATH_train)
    emb = torch.load(PATH_emb)
    # max_ans = max(map(lambda x: max(map(len, x[1])), train))
    # filt = filter(lambda a: a>13000, [len(x[1]) for x in train])
    # for i in filt:
    #     print(i)
    # a = answer_padding(train[0][1][0], emb, max_ans)
    # print(summ[0])

