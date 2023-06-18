import torch

class Optimiser():

    def __init__(self, m, lr):
        self.m = m
        self.optim = torch.optim.AdamW(m.parameters(), lr)

    def optimise(self, xb, yb):
        logits, loss = self.m(xb, yb)

        self.optim.zero_grad(set_to_none=True)

        loss.backward()

        self.optim.step()

        return logits, loss
