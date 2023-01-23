class Corpus:
    def __init__(self, corpus_file):
        with open(corpus_file) as f:
            self.corpus = [w.strip() for w in f]
        self.voc2idx = {word: idx for idx, word in enumerate(self.corpus)}
        self.corpus_size = len(self.corpus)

    def word2idx(self, word):
        return self.voc2idx[word] if word in self.voc2idx.keys() else self.voc2idx['<unk>']

    def idx2word(self, idx):
        return self.corpus[idx]