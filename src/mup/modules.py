import math
import einops
from numpy import sqrt
import torch as t
import torch.nn as nn
from params import Config

device = t.device('mps' if t.backends.mps.is_available() else 'cuda' if t.cuda.is_available() else 'cpu')


class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model), requires_grad=not cfg.fix_layernorm)
        self.b = nn.Parameter(t.zeros(cfg.d_model), requires_grad=not cfg.fix_layernorm)

    def forward(self, residual):
        if self.cfg.fix_layernorm:
            # Fix from u-mup https://arxiv.org/pdf/2407.17465
            # Use RMSNorm instead of LayerNorm, like LLaMA. Also, discard the gain parameter.
            residual_sq_mean = residual.square().mean(dim=-1, keepdim=True)
            return residual / (residual_sq_mean.sqrt() + self.cfg.layer_norm_eps)
        else:
            residual_mean = residual.mean(dim=-1, keepdim=True)
            residual_std = (residual.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps).sqrt()
            residual = (residual - residual_mean) / residual_std
            # print("wb means:", self.w.mean().item(), self.b.mean().item())
            return residual * self.w + self.b

class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=cfg.init_std)  # Input initialization scaling

    def forward(self, tokens):
        return self.W_E[tokens]

class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=cfg.init_std)  # Input initialization scaling

    def forward(self, tokens):
        batch, seq_len = tokens.shape
        return einops.repeat(self.W_pos[:seq_len], "seq d_model -> batch seq d_model", batch=batch)

class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Initialize weights with std=1 for hidden layers
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))

        if self.cfg.apply_muP:
            attn_init_std = cfg.init_std / math.sqrt(cfg.muP_width_multiplier)
            proj_init_std = cfg.init_std / math.sqrt(2 * cfg.n_layers * cfg.muP_width_multiplier)
        else:
            attn_init_std = cfg.init_std
            proj_init_std = cfg.init_std / math.sqrt(2 * cfg.n_layers)
        
        nn.init.normal_(self.W_Q, std=attn_init_std)
        nn.init.normal_(self.W_K, std=attn_init_std)
        nn.init.normal_(self.W_V, std=attn_init_std)
        nn.init.normal_(self.W_O, std=proj_init_std)

        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))

        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=device))

    def forward(self, normalized_resid_pre):
        # Compute query, key, and value vectors with activation scaling

        q = (einops.einsum(
            normalized_resid_pre, self.W_Q,
            "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
        ) + self.b_Q)

        k = (einops.einsum(
            normalized_resid_pre, self.W_K,
            "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
        ) + self.b_K)

        v = (einops.einsum(
            normalized_resid_pre, self.W_V,
            "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
        ) + self.b_V)

        # Compute attention scores and apply mask
        attn_scores = einops.einsum(
            q, k,
            "batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K",
        )
        
        if self.cfg.apply_muP:
            attn_scores /= self.cfg.d_head
        else:
            attn_scores /= math.sqrt(self.cfg.d_head)

        attn_scores_masked = self.apply_causal_mask(attn_scores)

        attn_pattern = attn_scores_masked.softmax(-1)

        # Compute attention output
        z = einops.einsum(
            attn_pattern, v,
            "batch nheads posn_Q posn_K, batch posn_K nheads d_head -> batch posn_Q nheads d_head",
        )
        attn_out = (einops.einsum(
            z, self.W_O,
            "batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model",
        ) + self.b_O) # Activation scaling

        return attn_out

    def apply_causal_mask(self, attn_scores):
        # Apply causal mask to attention scores
        mask = t.triu(t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores

class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        if self.cfg.apply_muP:
            nn.init.normal_(self.W_in, std=cfg.init_std / math.sqrt(cfg.muP_width_multiplier))
            nn.init.normal_(self.W_out, std=cfg.init_std / math.sqrt(2 * cfg.n_layers * cfg.muP_width_multiplier))
        else:
            nn.init.normal_(self.W_in, std=cfg.init_std)
            nn.init.normal_(self.W_out, std=cfg.init_std / math.sqrt(2 * cfg.n_layers))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))

    def forward(self, normalized_resid_mid):
        # Input to hidden layer with activation scaling
        pre = (einops.einsum(
            normalized_resid_mid, self.W_in,
            "batch position d_model, d_model d_mlp -> batch position d_mlp",
        ) + self.b_in)

        post = t.nn.functional.relu(pre)  # Using ReLU activation function

        mlp_out = (einops.einsum(
            post, self.W_out,
            "batch position d_mlp, d_mlp d_model -> batch position d_model",
        ) + self.b_out)

        return mlp_out

class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, resid_pre):
        attn_out = self.attn(self.ln1(resid_pre))
        resid_mid = resid_pre + attn_out

        mlp_out = self.mlp(self.ln2(resid_mid))
        resid_post = resid_mid + mlp_out

        return resid_post

class Unembed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_std)  # Standard initialization
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab)), requires_grad=False)

    def forward(self, normalized_resid_final):

        logits = normalized_resid_final @ self.W_U + self.b_U
        if self.cfg.fix_unembed:
            logits = logits / self.cfg.muP_width_multiplier

        return logits
    
class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)
    
    def init_weights(self):
        pass

    def configure_optimizers(self, weight_decay, learning_rate, betas=(0.9, 0.999), fix_embed_lr=False):
        if self.cfg.apply_muP:
            mup_decay_params = []
            decay_params = []
            nodecay_params = []

            embed_lr = learning_rate / sqrt(self.cfg.muP_width_multiplier) if fix_embed_lr else learning_rate / self.cfg.muP_width_multiplier
            decay_params.extend([self.embed.W_E, self.pos_embed.W_pos, self.unembed.W_U])
            for block in self.blocks:
                mup_decay_params.extend([
                     block.attn.W_Q, block.attn.W_K, block.attn.W_V, block.attn.W_O,
                ])
                mup_decay_params.extend([block.mlp.W_in, block.mlp.W_out])

                nodecay_params.extend([block.attn.b_Q, block.attn.b_K, block.attn.b_V, block.attn.b_O])
                nodecay_params.extend([block.mlp.b_in, block.mlp.b_out])
                nodecay_params.extend([block.ln1.w, block.ln1.b, block.ln2.w, block.ln2.b])

            nodecay_params.extend([self.ln_final.w, self.ln_final.b])

            optim_groups = [
                {'params': mup_decay_params, 'weight_decay': weight_decay, 'lr': learning_rate/self.cfg.muP_width_multiplier},
                {'params': decay_params, 'weight_decay': weight_decay, 'lr': embed_lr},
                {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate}
            ]
            optimizer = t.optim.AdamW(optim_groups, betas=betas)
        else:
            optimizer = t.optim.AdamW(self.parameters(),weight_decay=weight_decay, lr=learning_rate)

        return optimizer
    
    def forward(self, tokens):
        residual = self.embed(tokens) + self.pos_embed(tokens)

        if self.cfg.apply_muP:
            residual *= self.cfg.mu_input_alpha
        
        for block in self.blocks:
            residual = block(residual)
        
        residual = self.ln_final(residual)

        if self.cfg.apply_muP:
            residual *= self.cfg.mu_output_alpha / self.cfg.muP_width_multiplier
        
        logits = self.unembed(residual)
        return logits