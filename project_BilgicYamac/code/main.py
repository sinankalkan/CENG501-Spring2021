from BiLSTM import BiLSTM
from LoadData import LoadData
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
                         2,
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
            for i in range(len(answer)):
                try:
                    answer_matrix.append(emb[answer[i].item()])
                except:
                    answer_matrix.append(np.zeros(100))
            for i in range(13805 - len(answer)):
                answer_matrix.append(np.zeros(100))
            question_tensor = torch.FloatTensor(question_matrix)
            question_tensor = torch.reshape(question_tensor, (1, 22, 100))
            answer_tensor = torch.FloatTensor(answer_matrix)
            answer_tensor = torch.reshape(answer_tensor, (1, 13805, 100))
            # input_tensor = torch.cat((question_tensor, answer_tensor), dim=1)
            H_q = lstm.forward(question_tensor)
            H_a = lstm.forward(answer_tensor)




def main():
    train_QASelection()


if __name__ == '__main__':
    main()
