# from BiLSTM import BiLSTM
from LoadData import LoadData
from seq2seq import seq2seq
import torch as torch
import numpy as np


def get_epochs(data, numberOfParts):
    avg = len(data) / float(numberOfParts)
    out = []
    last = 0.0

    while last < len(data):
        out.append(data[int(last):int(last + avg)])
        last += avg

    return out


def train_QASelection(epochs=5):
    ld = LoadData()
    vocab = ld.loadVocab()
    ivocab = ld.loadIVocab()

    emb = ld.loadEmbeddings()
    train = ld.loadData("train")
    train_epochs = get_epochs(train, epochs)

    lstm = torch.nn.LSTM(100,
                         150,
                         bidirectional=True)

    for train_epoch in train_epochs:
        for train_input in train_epoch:
            question = train_input[0]
            answer = train_input[1]
            question_matrix = []
            answer_matrix = []
            for i in range(len(question)):
                try:
                    question_matrix.append(emb[question[i].item()])
                except Exception as e:
                    question_matrix.append(np.zeros(100))
            for i in range(22 - len(question)):
                question_matrix.append(np.zeros(100))
            for i in range(min(len(answer), 400)):
                try:
                    answer_matrix.append(emb[answer[i].item()])
                except:
                    answer_matrix.append(np.zeros(100))
            for i in range(400 - len(answer)):
                answer_matrix.append(np.zeros(100))
            question_tensor = torch.FloatTensor(question_matrix)
            question_tensor = torch.reshape(question_tensor, (1, 22, 100))
            answer_tensor = torch.FloatTensor(answer_matrix)
            answer_tensor = torch.reshape(answer_tensor, (1, 400, 100))
            H_q, c_q = lstm.forward(question_tensor)
            H_a, c_a = lstm.forward(answer_tensor)

            model = seq2seq(H_a.shape[2])
            a = model(H_a, H_q)


def main():
    train_QASelection()


if __name__ == '__main__':
    main()
