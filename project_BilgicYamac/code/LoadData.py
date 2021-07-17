from pathlib import Path

import torch as torch
import numpy as np


class LoadData:
    def __init__(self):
        my_file = Path("data/WikiQACorpus/vocab.t7")
        if not my_file.is_file():
            self.buildVocab()

        my_file = Path("data/WikiQACorpus/train.t7")
        if not my_file.is_file():
            self.buildData()
        pass

    def buildVocab(self):
        print("Building vocab dict ...")
        vocab = {}
        ivocab = []
        filenames = {"WikiQACorpus/WikiQACorpus/WikiQA-dev.txt",
                     "WikiQACorpus/WikiQACorpus/WikiQA-test.txt",
                     "WikiQACorpus/WikiQACorpus/WikiQA-train.txt"}
        for filename in filenames:
            with open(filename, encoding="utf8") as f:
                line = f.readline()
                while line:
                    divs = line.split("\t")
                    for m in range(0, 2):
                        words = divs[m].lower().split(" ")
                        for i in range(len(words)):
                            if words[i] not in vocab:
                                vocab[words[i]] = len(ivocab)
                                ivocab.append(words[i])

                    line = f.readline()

        torch.save(vocab, "data/WikiQACorpus/vocab.t7")
        torch.save(ivocab, "data/WikiQACorpus/ivocab.t7")

    def buildData(self):
        vocab = self.loadVocab()
        print("Building data ...")

        filenames = {("dev", "WikiQACorpus/WikiQACorpus/WikiQA-dev.txt"),
                     ("test", "WikiQACorpus/WikiQACorpus/WikiQA-test.txt"),
                     ("train", "WikiQACorpus/WikiQACorpus/WikiQA-train.txt")}

        for folder, filename in filenames:
            candidates = []
            labels = []
            instance = []
            lastQuestion = ""
            data = []
            with open(filename, encoding="utf8") as f:
                line = f.readline()
                while line:
                    divs = line.lower().split("\t")
                    if lastQuestion != divs[0]:
                        labels = torch.FloatTensor(labels)
                        if sum(labels) != 0:
                            words = lastQuestion.split(' ')
                            nwords = []
                            for word in words:
                                nwords.append(vocab[word])
                            instance.append(torch.LongTensor(nwords))
                            instance.append(candidates)
                            temp = np.array(labels)
                            instance.append(temp / sum(labels))
                            data.append(instance)

                        lastQuestion = divs[0]
                        instance = []
                        candidates = []
                        labels = []

                    words = divs[1].split(' ')
                    cand = []
                    for word in words:
                        cand.append(vocab[word])
                    candidates.append(torch.LongTensor(cand))
                    labels.append(int(divs[2]))
                    line = f.readline()
            torch.save(data, "data/sequence/" + folder + "_wikia.t7")
        return data

    def loadVocab(self):
        return torch.load("data/WikiQACorpus/vocab.t7")

    def loadIVocab(self):
        return torch.load("data/WikiQACorpus/ivocab.t7")

    def loadData(self, setname):
        return torch.load("data/sequence/" + setname + "_wikia.t7")
