from pathlib import Path

import torch as torch
import numpy as np


class LoadData:
    def __init__(self):
        my_file = Path("data/sequence/vocab.t7")
        if not my_file.is_file():
            self.buildVocab()

        my_file = Path("data/sequence/train_wikia.t7")
        if not my_file.is_file():
            self.buildSummary()
            self.buildData()

        my_file = Path("data/sequence/initEmb.t7")
        if not my_file.is_file():
            self.buildVacab2Emb()

    def buildVocab(self):
        print("Building vocab dict ...")
        vocab = {}
        ivocab = []
        filenames = {"WikiHowQACorpus/test.txt",
                     "WikiHowQACorpus/train.txt",
                     "WikiHowQACorpus/valid.txt"
                     }
        for filename in filenames:
            with open(filename, encoding="utf8") as f:
                line = f.readline()
                while line:
                    divs = line.split("\t")
                    words = divs[0].lower().split(" ")
                    for i in range(len(words)):
                        if words[i] not in vocab:
                            vocab[words[i]] = len(ivocab)
                            ivocab.append(words[i])

                    line = f.readline()

        with open("WikiHowQACorpus/summary.txt", encoding="utf8") as f:
            line = f.readline()
            while line:
                divs = line.split("\t")
                words = divs[1].lower().split(" ")
                for i in range(len(words)):
                    if words[i] not in vocab:
                        vocab[words[i]] = len(ivocab)
                        ivocab.append(words[i])

                line = f.readline()

        torch.save(vocab, "data/sequence/vocab.t7")
        torch.save(ivocab, "data/sequence/ivocab.t7")

    def buildData(self):
        vocab = self.loadVocab()
        summary = self.loadSummary()
        print("Building data ...")

        filenames = {("valid", "WikiHowQACorpus/valid.txt"),
                     ("test", "WikiHowQACorpus/test.txt"),
                     ("train", "WikiHowQACorpus/train.txt")}

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

                    summary_index = int(divs[1])
                    candidates.append(summary[summary_index])
                    labels.append(int(divs[2]))
                    line = f.readline()
            torch.save(data, "data/sequence/" + folder + "_wikia.t7")
        return data

    def buildSummary(self):
        vocab = self.loadVocab()
        print("Building summary ...")

        summaryMap = []
        with open("WikiHowQACorpus/summary.txt", encoding="utf8") as f:
            line = f.readline()
            line = line.replace("\n", " ")
            while line:
                divs = line.lower().split("\t")
                words = divs[1].split(' ')
                summary = []
                for word in words:
                    summary.append(vocab[word])
                summaryMap.append(torch.LongTensor(summary))
                line = f.readline()
        torch.save(summaryMap, "data/sequence/summary_wikia.t7")

    def buildVacab2Emb(self):
        vocab = self.loadVocab()
        print("Building embedings ...")

        emb = {}
        with open("glove/glove.6B.100d.txt", encoding="utf8") as f:
            line = f.readline()
            while line:
                vector = []
                vals = line.split("\n")[0].split(" ")
                if vals[0] in vocab:
                    for i in range(1, len(vals)):
                        vector.append(vals[i])
                    emb[vocab[vals[0]]]=vector
                line = f.readline()
        torch.save(emb, "data/sequence/initEmb.t7")

    def loadVocab(self):
        return torch.load("data/sequence/vocab.t7")

    def loadIVocab(self):
        return torch.load("data/sequence/ivocab.t7")

    def loadSummary(self):
        return torch.load("data/sequence/summary_wikia.t7")

    def loadEmbeddings(self):
        return torch.load("data/sequence/initEmb.t7")

    def loadData(self, setname):
        return torch.load("data/sequence/" + setname + "_wikia.t7")
