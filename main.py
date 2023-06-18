import torch
from loader import Loader
from tokeniser import Tokeniser
from batch import Batch
from bigram import BigramLanguageModel
from optimiser import Optimiser

BLOCK_SIZE = 512
BATCH_SIZE = 64
EMBEDDING_SIZE = 256
HEAD_SIZE = 64

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    l = Loader('./data/input.txt')
    tokeniser = Tokeniser(l.get_data())

    enc = tokeniser.encode()
    print(enc[:100])

    dec = tokeniser.decode(enc[:100])
    print(dec)

    batch = Batch(enc, batch_size=BATCH_SIZE, block_size=BLOCK_SIZE, train_split=0.9)

    xb, yb = batch.get_batch('train')

    print(xb, yb)

    m = BigramLanguageModel(block_size=BLOCK_SIZE, vocab_size=tokeniser.vocab_size, embedding_size=EMBEDDING_SIZE, head_size=HEAD_SIZE)
    print(m)
    logits, loss = m(xb, yb)

    print(logits.shape)
    print(loss)

    idx_new = torch.zeros((1,1), dtype=torch.long)
    idx_new = m.generate(idx_new, max_new_tokens=100)[0]
    # print(idx.tolist())
    print(tokeniser.decode(idx_new))

    o = Optimiser(m, 10e-4)

    for i in range(20000):
        xb, yb = batch.get_batch('train')
        logits, loss = o.optimise(xb, yb)
    
        # print(logits.shape)
        if i % 500 == 0:
            print('i: ' + str(i) + ' Loss: ' + str(loss.item()))
    print(loss.item())

    idx_new = torch.zeros((1,1), dtype=torch.long)
    idx_new = m.generate(idx_new, max_new_tokens=100)[0]
    # print(idx.tolist())
    print(tokeniser.decode(idx_new))

if __name__ == '__main__':
    main()