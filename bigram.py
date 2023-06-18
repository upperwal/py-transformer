import torch
import torch.nn as nn
from torch.nn import functional as F

from .attention import DecoderBlock

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, decoder_layer_size, block_size, embedding_size, head_size, dropout, device='cpu'):
        super().__init__()

        self.device = device

        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size) # 
        self.position_embedding_table = nn.Embedding(block_size, embedding_size)
        self.decoders = nn.Sequential(*[DecoderBlock(block_size, embedding_size, embedding_size // head_size, dropout) for _ in range(decoder_layer_size)])
        self.lm_head = nn.Linear(embedding_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx: B, T
        B, T = idx.shape

        # Look into the embedding table
        token_emb = self.token_embedding_table(idx) # B, T, C
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # T, C
        x = token_emb + pos_emb # B, T, C
        x = self.decoders(x) # B, T, C
        logits = self.lm_head(x) # B, T, V

        if targets == None:
            loss = None
        else:
            B, T, V = logits.shape
            logits = logits.view(B*T, V)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            
            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1) 

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, idx_next], dim=1)

        return idx
    

