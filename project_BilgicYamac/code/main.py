from LoadData import LoadData


def main():
    ld = LoadData()
    vocab = ld.loadVocab()
    ivocab = ld.loadIVocab()

    test = ld.loadData("test")
    train = ld.loadData("train")
    dev = ld.loadData("valid")
    emb = ld.loadEmbeddings()


if __name__ == '__main__':
    main()
