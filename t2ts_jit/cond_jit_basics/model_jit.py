# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import torch
import torch.nn as nn
import math
import os
import torch.nn.functional as F
from model_util import VisionRotaryEmbeddingFast, RotaryEmbedding1D, get_2d_sincos_pos_embed, get_1d_sincos_pos_embed, RMSNorm


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def maybe_compile(fn):
    # TorchInductor/Triton JIT can fail on some clusters due to missing CUDA toolchain
    # linkage (for example, libcuda during runtime C-extension build). Keep it opt-in.
    use_compile = os.environ.get("DRIFT_ENABLE_TORCH_COMPILE", "0") == "1"
    if use_compile and hasattr(torch, "compile"):
        return torch.compile(fn)
    return fn


class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class BottleneckPatchEmbed1D(nn.Module):
    def __init__(self, seq_len=256, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        num_patches = seq_len // patch_size
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv1d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv1d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, L = x.shape
        assert L == self.seq_len, f"Input length ({L}) doesn't match model ({self.seq_len})."
        x = self.proj2(self.proj1(x)).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype).to(device=query.device)

    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """
    The final layer of JiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    @maybe_compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class FinalLayer1D(nn.Module):
    """
    The final layer of JiT1D.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    @maybe_compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    @maybe_compile
    def forward(self, x,  c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class JiT(nn.Module):
    """
    Just image Transformer.
    """
    def __init__(
        self,
        text_embed_dim: int,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        bottleneck_dim=128,
        num_register_tokens=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size

        # time and cond embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        mid_size= (text_embed_dim + hidden_size) // 2
        self.text_embedder = nn.Sequential(
            nn.Linear(text_embed_dim, mid_size, bias=True),
            nn.GELU(),
            nn.Linear(mid_size, hidden_size, bias=True),
        )

        # linear embed
        self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)

        # use fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # register tokens (identity RoPE via num_cls_token — no natural 2D coords for these)
        self.num_register_tokens = num_register_tokens
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.randn(1, num_register_tokens, hidden_size) * 0.02)

        # rope: register positions get identity rotation (cos=1, sin=0)
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=num_register_tokens,
        )

        # transformer
        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        # linear predict
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)


        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, text_embeds):
        """
        x: (N, C, H, W)
        t: (N,)
        text_embeds: torch.Tensor
        """
        B = x.shape[0]
        t_emb = self.t_embedder(t)
        text_emb = self.text_embedder(text_embeds)
        c = t_emb + text_emb

        x = self.x_embedder(x)
        x += self.pos_embed

        if self.num_register_tokens > 0:
            register = self.register_tokens.expand(B, -1, -1)
            x = torch.cat([register, x], dim=1)

        for block in self.blocks:
            x = block(x, c, self.feat_rope)

        if self.num_register_tokens > 0:
            x = x[:, self.num_register_tokens:, :]

        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)
        return output


class JiT1D(nn.Module):
    def __init__(
        self,
        text_embed_dim: int,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        bottleneck_dim=128,
        num_register_tokens=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size

        # time and cond embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        mid_size = (text_embed_dim + hidden_size) // 2
        self.text_embedder = nn.Sequential(
            nn.Linear(text_embed_dim, mid_size, bias=True),
            nn.GELU(),
            nn.Linear(mid_size, hidden_size, bias=True),
        )

        # 1D patch embed
        self.x_embedder = BottleneckPatchEmbed1D(input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)

        # 1D sincos pos embed (patch tokens only)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # register tokens
        self.num_register_tokens = num_register_tokens
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.randn(1, num_register_tokens, hidden_size) * 0.02)

        # 1D RoPE covering register tokens + patch tokens with actual sequential positions
        head_dim = hidden_size // num_heads
        total_seq_len = num_register_tokens + num_patches
        self.feat_rope = RotaryEmbedding1D(dim=head_dim, seq_len=total_seq_len)

        # transformer
        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        # 1D final layer
        self.final_layer = FinalLayer1D(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 1D sincos pos embed
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.num_patches)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize Conv1d patch embed weights
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        """
        x: (N, T, patch_size * C)
        out: (N, C, L)
        """
        c = self.out_channels
        T = x.shape[1]
        x = x.reshape(x.shape[0], T, p, c)
        x = x.permute(0, 3, 1, 2)
        return x.reshape(x.shape[0], c, T * p)

    def forward(self, x, t, text_embeds):
        """
        x: (N, C, L)
        t: (N,)
        text_embeds: torch.Tensor
        """
        B = x.shape[0]
        t_emb = self.t_embedder(t)
        text_emb = self.text_embedder(text_embeds)
        c = t_emb + text_emb

        x = self.x_embedder(x)
        x += self.pos_embed

        if self.num_register_tokens > 0:
            register = self.register_tokens.expand(B, -1, -1)
            x = torch.cat([register, x], dim=1)

        for block in self.blocks:
            x = block(x, c, self.feat_rope)

        if self.num_register_tokens > 0:
            x = x[:, self.num_register_tokens:, :]

        x = self.final_layer(x, c)
        return self.unpatchify(x, self.patch_size)


class JiTBlockMultiCond(nn.Module):
    """JiTBlock variant that accepts per-scale conditioning ``c`` of shape (B, K_slot, H).

    Each c slot owns one contiguous chunk of tokens along the sequence axis;
    lengths are given by ``scale_lengths`` and must sum to ``x.shape[1]``.
    adaLN-Zero produces 6 modulation parameters per slot; each is broadcast
    along its slot's tokens so all tokens in a slot share the same shift /
    scale / gate. Attention runs across the full concatenated sequence.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    @staticmethod
    def _expand_per_scale(t, scale_lengths):
        # t: (B, K_slot, H) -> (B, sum(scale_lengths), H)
        return torch.cat(
            [t[:, i:i + 1, :].expand(-1, L_i, -1) for i, L_i in enumerate(scale_lengths)],
            dim=1,
        )

    def forward(self, x, c, scale_lengths, feat_rope=None):
        # x: (B, N, H); c: (B, K_slot, H); scale_lengths: list[int] summing to N
        ada = self.adaLN_modulation(c)  # (B, K_slot, 6H)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada.chunk(6, dim=-1)
        sh_a = self._expand_per_scale(shift_msa, scale_lengths)
        sc_a = self._expand_per_scale(scale_msa, scale_lengths)
        ga_a = self._expand_per_scale(gate_msa, scale_lengths)
        sh_m = self._expand_per_scale(shift_mlp, scale_lengths)
        sc_m = self._expand_per_scale(scale_mlp, scale_lengths)
        ga_m = self._expand_per_scale(gate_mlp, scale_lengths)

        x = x + ga_a * self.attn(self.norm1(x) * (1 + sc_a) + sh_a, rope=feat_rope)
        x = x + ga_m * self.mlp(self.norm2(x) * (1 + sc_m) + sh_m)
        return x


class MultiResolutionJiT1D(nn.Module):
    """Multi-resolution 1D JiT with per-scale text conditioning.

    Differences from ``JiT1D``:
    - Input is patched at K = ``len(patch_sizes)`` different resolutions in
      parallel (one ``BottleneckPatchEmbed1D`` per scale, all sharing
      ``bottleneck_dim``). Each scale gets its own fixed sin-cos positional
      embedding.
    - Text condition is K tokens (one per scale); each scale's tokens see
      adaLN modulation derived from its own text token plus the (global)
      time embedding. Attention is shared across the full multi-scale
      sequence, and RoPE runs over the concatenated positions.
    - Output is decoded per scale (per-scale ``FinalLayer1D`` + unpatchify),
      then merged via a Linear(K, 1) mixer (VerbalTS-style).
    """
    def __init__(
        self,
        text_embed_dim: int,
        input_size: int = 256,
        patch_sizes: list[int] | None = None,
        in_channels: int = 3,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bottleneck_dim: int = 128,
        num_register_tokens: int = 0,
    ):
        super().__init__()
        assert patch_sizes is not None and len(patch_sizes) > 0, "patch_sizes must be a non-empty list"
        for p in patch_sizes:
            assert input_size % p == 0, f"input_size={input_size} must be divisible by patch_size={p}"

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_sizes = list(patch_sizes)
        self.multipatch_num = len(self.patch_sizes)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size

        # time embedder (global, shared across scales)
        self.t_embedder = TimestepEmbedder(hidden_size)

        # text embedder: shared MLP applied per-token to (B, K, text_embed_dim)
        mid_size = (text_embed_dim + hidden_size) // 2
        self.text_embedder = nn.Sequential(
            nn.Linear(text_embed_dim, mid_size, bias=True),
            nn.GELU(),
            nn.Linear(mid_size, hidden_size, bias=True),
        )

        # one BottleneckPatchEmbed1D per scale (shared bottleneck_dim)
        self.x_embedders = nn.ModuleList([
            BottleneckPatchEmbed1D(input_size, p, in_channels, bottleneck_dim, hidden_size, bias=True)
            for p in self.patch_sizes
        ])
        self.num_patches_per_scale = [input_size // p for p in self.patch_sizes]

        # per-scale fixed sin-cos positional embeddings
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, n_p, hidden_size), requires_grad=False)
            for n_p in self.num_patches_per_scale
        ])

        # register tokens (optional)
        self.num_register_tokens = num_register_tokens
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.randn(1, num_register_tokens, hidden_size) * 0.02)

        # scale_lengths: register slot first (if any), then K patch scales.
        # This is fixed at construction time and reused by every block.
        scale_lengths = []
        if num_register_tokens > 0:
            scale_lengths.append(num_register_tokens)
        scale_lengths.extend(self.num_patches_per_scale)
        self.scale_lengths = scale_lengths
        self.num_c_slots = len(scale_lengths)

        # global 1D RoPE over the full concatenated sequence
        head_dim = hidden_size // num_heads
        total_seq_len = sum(scale_lengths)
        self.feat_rope = RotaryEmbedding1D(dim=head_dim, seq_len=total_seq_len)

        # transformer (multi-condition variant)
        self.blocks = nn.ModuleList([
            JiTBlockMultiCond(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
            )
            for i in range(depth)
        ])

        # per-scale final layers + multi-patch merge
        self.final_layers = nn.ModuleList([
            FinalLayer1D(hidden_size, p, self.out_channels)
            for p in self.patch_sizes
        ])
        self.multipatch_mixer = nn.Linear(self.multipatch_num, 1, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # per-scale 1D sincos pos embeds
        for i, pos_embed in enumerate(self.pos_embeds):
            pe = get_1d_sincos_pos_embed(pos_embed.shape[-1], self.num_patches_per_scale[i])
            pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

        # init each patch embedder's Conv1d weights
        for emb in self.x_embedders:
            w1 = emb.proj1.weight.data
            nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
            w2 = emb.proj2.weight.data
            nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
            nn.init.constant_(emb.proj2.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # zero-init adaLN modulation in transformer blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # zero-init each final layer (adaLN-Zero)
        for fl in self.final_layers:
            nn.init.constant_(fl.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(fl.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(fl.linear.weight, 0)
            nn.init.constant_(fl.linear.bias, 0)
        # multipatch_mixer keeps its xavier init from _basic_init.

    @staticmethod
    def _unpatchify_1d(x, p, out_channels):
        # x: (B, T, p * C) -> (B, C, T * p)
        T = x.shape[1]
        x = x.reshape(x.shape[0], T, p, out_channels)
        x = x.permute(0, 3, 1, 2)
        return x.reshape(x.shape[0], out_channels, T * p)

    def forward(self, x, t, text_embeds):
        """
        x:           (B, in_channels, L)        with L == input_size
        t:           (B,)
        text_embeds: (B, K, text_embed_dim)     with K == multipatch_num
        returns:     (B, out_channels, L)
        """
        B = x.shape[0]
        assert text_embeds.dim() == 3, "text_embeds must have shape (B, K, text_embed_dim)"
        assert text_embeds.shape[1] == self.multipatch_num, (
            f"text_embeds K={text_embeds.shape[1]} must match multipatch_num={self.multipatch_num}"
        )

        t_emb = self.t_embedder(t)                                  # (B, H)
        text_emb = self.text_embedder(text_embeds)                  # (B, K, H)

        # per-scale c: each scale sees t_emb + its own text token
        c_scales = t_emb.unsqueeze(1) + text_emb                    # (B, K, H)
        if self.num_register_tokens > 0:
            c_register = (t_emb + text_emb.mean(dim=1)).unsqueeze(1)
            c = torch.cat([c_register, c_scales], dim=1)            # (B, K+1, H)
        else:
            c = c_scales                                            # (B, K, H)

        # multi-patch encode + per-scale positional embeddings
        x_parts = []
        for i in range(self.multipatch_num):
            xi = self.x_embedders[i](x)                             # (B, n_p_i, H)
            xi = xi + self.pos_embeds[i]
            x_parts.append(xi)
        x_concat = torch.cat(x_parts, dim=1)                        # (B, sum_n_p, H)

        if self.num_register_tokens > 0:
            register = self.register_tokens.expand(B, -1, -1)
            x_concat = torch.cat([register, x_concat], dim=1)       # (B, n_reg + sum_n_p, H)

        for block in self.blocks:
            x_concat = block(x_concat, c, self.scale_lengths, self.feat_rope)

        if self.num_register_tokens > 0:
            x_concat = x_concat[:, self.num_register_tokens:, :]

        # multi-patch decode: per-scale final layer + unpatchify
        outs = []
        start = 0
        for i in range(self.multipatch_num):
            n_p = self.num_patches_per_scale[i]
            xi = x_concat[:, start:start + n_p, :]                  # (B, n_p, H)
            c_i = t_emb + text_emb[:, i, :]                         # (B, H), per-scale global c
            xi = self.final_layers[i](xi, c_i)                      # (B, n_p, p_i * out_channels)
            xi = self._unpatchify_1d(xi, self.patch_sizes[i], self.out_channels)  # (B, out_channels, L)
            outs.append(xi)
            start += n_p

        # merge: (B, C, L, K) -> Linear(K, 1) -> (B, C, L, 1) -> squeeze
        stacked = torch.stack(outs, dim=-1)                         # (B, C, L, K)
        merged = self.multipatch_mixer(stacked).squeeze(-1)         # (B, C, L)
        return merged
