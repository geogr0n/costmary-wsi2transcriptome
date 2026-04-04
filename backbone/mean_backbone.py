"""
Shallow mean-pool control:
x(B,T,D) -> mean over T -> prediction head.

This intentionally removes the intermediate stem MLP so the backbone is a
strict pooled-feature baseline.
"""
import torch
import torch.nn as nn

from heads import SPEXHead, make_layernorm_linear_head


class MeanBackbone(nn.Module):
    """x(B,T,D) -> mean over T -> head."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_outputs: int,
        spd_r: int = 256,
        spd_eps: float = 1e-3,
        spd_latent_dim: int = 256,
        spd_dropout: float = 0.1,
        spd_mlp_depth: int = 1,
        device: str | torch.device = "cuda:0",
        head_type: str = "linear",
        spex_k: int = 64,
        pca_mu=None,
        pca_U_k=None,
    ):
        super().__init__()
        self.device = device
        self.input_dim = int(input_dim)
        self.num_outputs = int(num_outputs)
        self.head_type = str(head_type).lower()

        # Keep SPD-shaped constructor args for compatibility with the shared
        # training entrypoint, but this baseline uses only direct mean pooling.
        _ = (spd_r, spd_eps, spd_latent_dim, spd_dropout, spd_mlp_depth)

        mu_t = None
        if pca_mu is not None:
            mu_t = torch.from_numpy(pca_mu).float() if not isinstance(pca_mu, torch.Tensor) else pca_mu.float()

        latent = self.input_dim
        use_spex = self.head_type == "spex"
        if use_spex and pca_mu is not None and pca_U_k is not None:
            uk_t = torch.from_numpy(pca_U_k).float() if not isinstance(pca_U_k, torch.Tensor) else pca_U_k.float()
            self.linear_head = SPEXHead(latent, self.num_outputs, spex_k, mu_t, uk_t)
        else:
            base_head = make_layernorm_linear_head(latent, self.num_outputs)
            if use_spex:
                self.linear_head = SPEXHead(latent, self.num_outputs, spex_k)
            else:
                self.linear_head = base_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            z = x.mean(dim=1)
        elif x.dim() == 2:
            z = x
        else:
            raise ValueError(f"Expected x shape (B,T,D) or (B,D), got {tuple(x.shape)}")
        return self.linear_head(z)
