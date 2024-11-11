from dataclasses import dataclass

@dataclass
class TransformerTrainingArgs:
    lr : float = 1e-3
    weight_decay : float = 1e-2
    betas : tuple[float,float] = (0.9,0.999)
    wandb_project: str | None = "day1-demotransformer"
    wandb_name: str | None = None
    batch_size : int = 16
    epochs: int= 20
    max_steps_per_epoch: int = -1
    collect_norms: bool = False
    

@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    apply_muP: bool = False
    d_vocab: int = 50257
    init_std: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    mu_output_alpha: float = 1.0
    mu_input_alpha: float = 1.0
    muP_width_multiplier: float = 1.0
    n_heads: int = 12
    n_layers: int = 12



