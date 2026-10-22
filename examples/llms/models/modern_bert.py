#!/usr/bin/env python3

import jax.numpy as jnp

import prix
import prix.nn as nn


class ModernBertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size, bias=False)

    def forward(self, input_ids, token_type_ids=None):
        return self.norm(self.tok_embeddings(input_ids))


class ModernBertRotaryEmbedding(nn.Module):
    def __init__(self, dim, base):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.size(1)
        position_ids_expanded = jnp.arange(0, seq_len, dtype=jnp.float32)[None, None, :]
        inv_freq_expanded = jnp.broadcast_to(
            self.inv_freq[None, :, None], (batch_size, self.inv_freq.shape[0], 1)
        )
        freqs = jnp.swapaxes(inv_freq_expanded @ position_ids_expanded, 1, 2)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return prix.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
    sin = jnp.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class ModernBertAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, layer_id):
        super().__init__()
        self.local_attention = 128
        self.global_attn_every_n_layers = 3
        self.hidden_size = hidden_size
        self.layer_id = layer_id

        if self.layer_id % self.global_attn_every_n_layers != 0:
            base = 10_000
        else:
            base = 160_000

        self.n_heads = n_heads
        self.Wqkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.head_dim = hidden_size // n_heads
        self.rotary_emb = ModernBertRotaryEmbedding(self.head_dim, base)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        qkv = self.Wqkv(hidden_states)
        cos, sin = self.rotary_emb(qkv)

        query, key, value = qkv.split(self.hidden_size, dim=2)
        query = query.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        attn = query.matmul(key.transpose(2, 3)) / (query.size(-1) ** 0.5)

        if self.layer_id % self.global_attn_every_n_layers != 0:
            rows = jnp.arange(attn.shape[2])[:, None]
            distance = jnp.abs(rows - rows.T)
            window_mask = (distance <= self.local_attention // 2)[None, None, :, :]
            attn.share = jnp.where(window_mask, attn.share, jnp.int64(-(2**46)))

        attn = attn.softmax(dim=-1)

        emb_rich = attn.matmul(value).transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        emb_rich = self.Wo(emb_rich)
        return emb_rich


class ModernBertMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.Wi = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.act = nn.GELU()
        self.Wo = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        input_state, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.act(input_state) * gate)


class ModernBertLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, n_heads, layer_id):
        super().__init__()
        if layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(hidden_size, bias=False)
        self.attn = ModernBertAttention(hidden_size, n_heads, layer_id)
        self.mlp_norm = nn.LayerNorm(hidden_size, bias=False)
        self.mlp = ModernBertMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states))
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class ModernBertModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        seq_len,
        n_heads,
        n_layers,
        full=True,
    ):
        super().__init__()
        self.full = full
        self.embed_dim = hidden_size
        if full:
            self.embeddings = ModernBertEmbeddings(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                ModernBertLayer(hidden_size, intermediate_size, n_heads, layer_id)
                for layer_id in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size, bias=False)

    def forward(self, input_ids, input_embeds=None):
        hidden_states = self.embeddings(input_ids, input_embeds) if self.full else input_ids
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        return hidden_states


class ModernBertPredictionHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size, bias=False)

    def forward(self, hidden_states):
        hidden_states = hidden_states[:, 0]
        out = self.norm(self.act(self.dense(hidden_states)))
        return out


class ModernBertForSequenceClassification(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        seq_len,
        n_heads,
        n_layers,
        classes=2,
    ):
        super().__init__()
        self.model = ModernBertModel(
            vocab_size, hidden_size, intermediate_size, seq_len, n_heads, n_layers
        )
        self.head = ModernBertPredictionHead(hidden_size)
        self.classifier = nn.Linear(hidden_size, classes)

    def forward(self, input_ids, token_type_ids=None):
        pooled_out = self.model(input_ids, token_type_ids)
        out = self.head(pooled_out)
        return self.classifier(out)


class ModernBert(ModernBertModel):
    def __init__(self, seq_len, full=True):
        super().__init__(
            vocab_size=50368,
            hidden_size=768,
            intermediate_size=1152,
            seq_len=seq_len,
            n_heads=12,
            n_layers=22,
            full=full,
        )


class ModernBertLarge(ModernBertModel):
    def __init__(self, seq_len, full=True):
        super().__init__(
            vocab_size=50368,
            hidden_size=1024,
            intermediate_size=2624,
            seq_len=seq_len,
            n_heads=16,
            n_layers=24,
            full=full,
        )


class ModernBertForTokenClassification(ModernBertForSequenceClassification):
    def __init__(self):
        super().__init__(
            vocab_size=50368,
            hidden_size=768,
            intermediate_size=1152,
            seq_len=8192,
            n_heads=12,
            n_layers=22,
        )


class ModernBertLargeForTokenClassification(ModernBertForSequenceClassification):
    def __init__(self):
        super().__init__(
            vocab_size=50368,
            hidden_size=1024,
            intermediate_size=2624,
            seq_len=8192,
            n_heads=16,
            n_layers=24,
            classes=2,
        )
