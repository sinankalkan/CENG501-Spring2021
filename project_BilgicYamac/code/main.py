from LoadData import LoadData


def main():
    ld = LoadData()
    vocab = ld.loadVocab()
    ivocab = ld.loadIVocab()

    test = ld.loadData("test")
    train = ld.loadData("train")
    dev = ld.loadData("dev")
    pass


if __name__ == '__main__':
    main()
