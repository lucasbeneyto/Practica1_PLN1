import torch
import torch.nn as nn

from .rope import compute_rope_params, apply_rope


# ============================================================
# Layer Normalization
# ============================================================

class LayerNorm(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()

        self.eps = 1e-5

        self.scale = nn.Parameter(
            torch.ones(emb_dim)
        )

        self.shift = nn.Parameter(
            torch.zeros(emb_dim)
        )

    def forward(self, x):

        mean = x.mean(
            dim=-1,
            keepdim=True
        )

        var = x.var(
            dim=-1,
            keepdim=True,
            unbiased=False
        )

        norm_x = (
            (x - mean)
            / torch.sqrt(var + self.eps)
        )

        return (
            self.scale * norm_x
            + self.shift
        )


# ============================================================
# GELU
# ============================================================

class GELU(nn.Module):

    def forward(self, x):

        return 0.5 * x * (
            1.0
            + torch.tanh(
                torch.sqrt(
                    torch.tensor(
                        2.0 / torch.pi,
                        device=x.device
                    )
                )
                * (
                    x
                    + 0.044715 * x ** 3
                )
            )
        )


# ============================================================
# Feed Forward
# ============================================================

class FeedForward(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(

            nn.Linear(
                cfg["emb_dim"],
                4 * cfg["emb_dim"]
            ),

            GELU(),

            nn.Linear(
                4 * cfg["emb_dim"],
                cfg["emb_dim"]
            )
        )

    def forward(self, x):

        return self.layers(x)


# ============================================================
# Multi-Head Attention
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout,
        num_heads,
        qkv_bias=False,
        use_rope=False
    ):
        super().__init__()

        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads

        self.head_dim = (
            d_out // num_heads
        )

        self.use_rope = use_rope

        # Proyecciones Q, K y V
        self.W_query = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias
        )

        self.W_key = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias
        )

        self.W_value = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias
        )

        # Proyección de salida
        self.out_proj = nn.Linear(
            d_out,
            d_out
        )

        self.dropout = nn.Dropout(
            dropout
        )

        # Máscara causal
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(
                    context_length,
                    context_length
                ),
                diagonal=1
            )
        )

        # ----------------------------------------------------
        # RoPE
        # ----------------------------------------------------

        if self.use_rope:

            cos, sin = compute_rope_params(
                head_dim=self.head_dim,
                context_length=context_length
            )

            self.register_buffer(
                "rope_cos",
                cos
            )

            self.register_buffer(
                "rope_sin",
                sin
            )


    def forward(self, x):

        b, num_tokens, d_in = x.shape

        # ----------------------------------------
        # 1. Proyecciones Q, K y V
        # ----------------------------------------

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # ----------------------------------------
        # 2. Separación en cabezas
        #
        # (B, T, D)
        #      ↓
        # (B, T, H, head_dim)
        # ----------------------------------------

        keys = keys.view(
            b,
            num_tokens,
            self.num_heads,
            self.head_dim
        )

        queries = queries.view(
            b,
            num_tokens,
            self.num_heads,
            self.head_dim
        )

        values = values.view(
            b,
            num_tokens,
            self.num_heads,
            self.head_dim
        )

        # ----------------------------------------
        # 3. Reordenamos
        #
        # (B, T, H, D)
        #      ↓
        # (B, H, T, D)
        # ----------------------------------------

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # ----------------------------------------
        # 4. RoPE
        #
        # SOLO Q y K
        # ----------------------------------------

        if self.use_rope:

            queries = apply_rope(
                queries,
                self.rope_cos,
                self.rope_sin
            )

            keys = apply_rope(
                keys,
                self.rope_cos,
                self.rope_sin
            )

        # ----------------------------------------
        # 5. Scores de atención
        # ----------------------------------------

        attn_scores = (
            queries
            @ keys.transpose(2, 3)
        )

        # Máscara causal
        mask_bool = (
            self.mask.bool()
            [:num_tokens, :num_tokens]
        )

        attn_scores.masked_fill_(
            mask_bool,
            -torch.inf
        )

        # ----------------------------------------
        # 6. Softmax
        # ----------------------------------------

        attn_weights = torch.softmax(
            attn_scores
            / self.head_dim ** 0.5,
            dim=-1
        )

        attn_weights = self.dropout(
            attn_weights
        )

        # ----------------------------------------
        # 7. Aplicamos atención a V
        # ----------------------------------------

        context_vec = (
            attn_weights @ values
        )

        # (B, H, T, D)
        #      ↓
        # (B, T, H, D)

        context_vec = context_vec.transpose(
            1,
            2
        )

        # Volvemos a juntar las cabezas

        context_vec = (
            context_vec
            .contiguous()
            .view(
                b,
                num_tokens,
                self.d_out
            )
        )

        # Proyección de salida

        context_vec = self.out_proj(
            context_vec
        )

        return context_vec


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        cfg,
        use_rope=False
    ):
        super().__init__()

        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            use_rope=use_rope
        )

        self.ff = FeedForward(cfg)

        self.norm1 = LayerNorm(
            cfg["emb_dim"]
        )

        self.norm2 = LayerNorm(
            cfg["emb_dim"]
        )

        self.drop_shortcut = nn.Dropout(
            cfg["drop_rate"]
        )


    def forward(self, x):

        # ----------------------------------------
        # Attention + residual connection
        # ----------------------------------------

        shortcut = x

        x = self.norm1(x)

        x = self.att(x)

        x = self.drop_shortcut(x)

        x = x + shortcut

        # ----------------------------------------
        # FeedForward + residual connection
        # ----------------------------------------

        shortcut = x

        x = self.norm2(x)

        x = self.ff(x)

        x = self.drop_shortcut(x)

        x = x + shortcut

        return x


# ============================================================
# GPT Model
# ============================================================

class GPTModel(nn.Module):

    def __init__(
        self,
        cfg,
        use_rope=False
    ):
        super().__init__()

        self.use_rope = use_rope

        # ----------------------------------------
        # Token embeddings
        # ----------------------------------------

        self.tok_emb = nn.Embedding(
            cfg["vocab_size"],
            cfg["emb_dim"]
        )

        # ----------------------------------------
        # Positional embeddings
        #
        # SOLO para baseline
        # ----------------------------------------

        if not self.use_rope:

            self.pos_emb = nn.Embedding(
                cfg["context_length"],
                cfg["emb_dim"]
            )

        # ----------------------------------------
        # Dropout
        # ----------------------------------------

        self.drop_emb = nn.Dropout(
            cfg["drop_rate"]
        )

        # ----------------------------------------
        # Transformer blocks
        # ----------------------------------------

        self.trf_blocks = nn.Sequential(

            *[
                TransformerBlock(
                    cfg,
                    use_rope=use_rope
                )

                for _ in range(
                    cfg["n_layers"]
                )
            ]
        )

        # ----------------------------------------
        # Final normalization
        # ----------------------------------------

        self.final_norm = LayerNorm(
            cfg["emb_dim"]
        )

        # ----------------------------------------
        # Vocabulary output
        # ----------------------------------------

        self.out_head = nn.Linear(
            cfg["emb_dim"],
            cfg["vocab_size"],
            bias=False
        )


    def forward(self, in_idx):

        batch_size, seq_len = (
            in_idx.shape
        )

        # ----------------------------------------
        # Token embeddings
        # ----------------------------------------

        tok_embeds = self.tok_emb(
            in_idx
        )

        # ----------------------------------------
        # Positional information
        # ----------------------------------------

        if self.use_rope:

            # RoPE se aplicará dentro
            # de cada Attention.
            x = tok_embeds

        else:

            positions = torch.arange(
                seq_len,
                device=in_idx.device
            )

            pos_embeds = self.pos_emb(
                positions
            )

            x = (
                tok_embeds
                + pos_embeds
            )

        # ----------------------------------------
        # GPT
        # ----------------------------------------

        x = self.drop_emb(x)

        x = self.trf_blocks(x)

        x = self.final_norm(x)

        logits = self.out_head(x)

        return logits