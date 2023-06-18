import torch

class Tokeniser():
    def __init__(self, data):
        self.data = data

        self._process()

    def _process(self):
        unique_chars = sorted(list(set(self.data)))
        self.vocab_size = len(unique_chars)

        self.stoi = {c:i for i, c in enumerate(unique_chars)}
        self.itos = {i:c for c, i in self.stoi.items()}
    
    def encode(self):
        arr = [self.stoi[c] for c in self.data]

        return torch.tensor(arr, dtype=torch.long)
    
    def decode(self, i_list):
        return ''.join([self.itos[i] for i in i_list.tolist()])


