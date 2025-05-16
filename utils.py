import re
from collections import Counter

class SimpleTokenizer:
    def __init__(self, captions, min_freq=5):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.build_vocab(captions, min_freq)

    def build_vocab(self, captions, min_freq):
        counter = Counter()
        for caption in captions:
            tokens = self.tokenize(caption)
            counter.update(tokens)
        for word, freq in counter.items():
            if freq >= min_freq:
                self.word2idx[word] = len(self.word2idx)

    def tokenize(self, text):
        return re.findall(r'\w+', text.lower())

    def encode(self, text, max_len=20):
        tokens = self.tokenize(text)
        token_ids = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]
        token_ids = token_ids[:max_len]
        token_ids += [self.word2idx["<PAD>"]] * (max_len - len(token_ids))
        return token_ids

    def __len__(self):
        return len(self.word2idx)
