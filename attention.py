import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):

    def __init__(self, block_size, embedding_size, no_of_heads):
        super().__init__()

        self.multiple_heads = MultiHeadAttention(no_of_heads, block_size, embedding_size, embedding_size//no_of_heads)
        self.layer_norm_attn = nn.LayerNorm(embedding_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU()
        )

        self.layer_norm_ffn = nn.LayerNorm(embedding_size)

    def forward(self, x):
        heads_out = self.multiple_heads(x)
        norm = self.layer_norm_attn(heads_out + x)
        ff = self.feed_forward(norm)
        return self.layer_norm_ffn(ff + norm + x)

class MultiHeadAttention(nn.Module):

    def __init__(self, no_heads, block_size, embedding_size, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(block_size, embedding_size, head_size) for _ in range(no_heads)])
        self.linear = nn.Linear(no_heads*head_size, embedding_size)


    def forward(self, x):
        cat = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.linear(cat)

class Head(nn.Module):

    def __init__(self, block_size, embedding_size, head_size):

        super().__init__()

        self.head_size = head_size
        
        self.key = nn.Linear(embedding_size, head_size, bias=False) # B, C, H
        self.query = nn.Linear(embedding_size, head_size, bias=False) # B, C, H
        self.value = nn.Linear(embedding_size, head_size, bias=False) # B, C, H

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):

        B, T, C = x.shape

        # x: B, T, C

        k = self.key(x) # B, T, H
        q = self.key(x) # B, T, H

        wei = q @ k.transpose(-2, -1) * self.head_size**-5 # B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        out = wei @ self.value(x) # B, T, H

        return out
        

