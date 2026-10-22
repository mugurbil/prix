#!/usr/bin/env python3

import jax.numpy as jnp

import prix.nn as nn


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, layer_id):
        super().__init__()
        self.attention = GPTAttention(embed_dim, num_heads, layer_id)

    def forward(self, x, mask=True):
        return self.attention(x, mask)


class GPTAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, layer_id):
        super().__init__()
        assert embed_dim % num_heads == 0, "invalid heads and embedding dimension"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.search_dim = embed_dim // num_heads
        self.layer_id = layer_id

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _split_heads(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.search_dim)
        x = x.view(new_shape)
        return x.permute((0, 2, 1, 3))

    def forward(self, x, mask=True):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        attn = query.matmul(key.transpose(-1, -2))
        if mask:
            if self.layer_id % 2 != 0:
                rows = jnp.arange(attn.shape[2])[:, None]
                distance = jnp.abs(rows - rows.T)
                window_mask = (distance <= 128)[None, None, :, :]
                attn.share = jnp.where(window_mask, attn.share, jnp.int64(-(2**46)))
            ones = jnp.ones_like(attn.share, dtype=jnp.int64)
            tril_mask = jnp.tril(ones, k=0)
            triu_mask = jnp.triu(ones, k=1)
            attn = attn * tril_mask
            attn.share = attn.share + jnp.int64(-(2**46)) * triu_mask
        attn = attn.softmax(dim=-1)

        y = attn.matmul(value).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        y = self.out_proj(y)
        return y


class GPTMLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, embed_dim * 4)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(embed_dim * 4, embed_dim)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x


class GPTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, layer_id):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, layer_id)
        self.mlp = GPTMLP(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_blocks, vocab_size, seq_len, full):
        super().__init__()
        self.full = full
        self.embed_dim = embed_dim
        self.h = nn.Sequential(
            *[GPTBlock(embed_dim, num_heads, layer_id) for layer_id in range(num_blocks)]
        )
        if full:
            self.wte = nn.Embedding(vocab_size, embed_dim)
            self.wpe = nn.Embedding(seq_len, embed_dim)
            self.ln_f = nn.LayerNorm(embed_dim)

    def forward(self, x, target=None):
        if self.full:
            tok_embedding = self.wte(x)
            pos_embedding = self.wpe.weight[: x.size()[1], :].reshape(1, x.size()[1], -1)
            x = tok_embedding + pos_embedding
        x = self.h(x)
        if self.full:
            x = self.ln_f(x)
        return x


class GPT(nn.Module):
    def __init__(self, embed_dim, num_heads, num_blocks, vocab_size, seq_len, full=True):
        super().__init__()
        self.full = full
        self.embed_dim = embed_dim
        self.transformer = Transformer(embed_dim, num_heads, num_blocks, vocab_size, seq_len, full)
        if full:
            self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x, target=None):
        x = self.transformer(x, target)
        if self.full:
            x = self.lm_head(x)
        return x


class GPTNeo(GPT):
    def __init__(self, seq_len=2048, full=True):
        super().__init__(
            embed_dim=2048,
            num_heads=16,
            num_blocks=24,
            vocab_size=50257,
            seq_len=seq_len,
            full=full,
        )


class GPTNeoLMHead(GPT):
    def __init__(self, seq_len=2048):
        super().__init__(
            embed_dim=2048,
            num_heads=16,
            num_blocks=24,
            vocab_size=50257,
            seq_len=seq_len,
        )
