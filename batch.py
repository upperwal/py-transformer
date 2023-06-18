import torch

class Batch():
    def __init__(self, data, batch_size, block_size, train_split=None):
        self.batch_size = batch_size
        self.block_size = block_size
        self.data = data
        self.train_split = train_split

        self._split()

    def _split(self):
        if self.train_split:
            t_len = int(self.train_split * len(self.data))
            self.train = self.data[:t_len]
            self.val = self.data[t_len:]

    def get_batch(self, split):
        data = self.train if split == 'train' else self.val
        idx = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in idx])
        y = torch.stack([data[i+1 : i + self.block_size+1] for i in idx])

        return x, y