import torch
import torch.nn as nn

from heads import SPEXHead, make_layernorm_linear_head


class ProjMeanBackbone(nn.Module):
    """
    First-order control for SPD:
    x(B,T,D) -> LayerNorm+Linear(D->r) -> mean over T -> bottleneck -> head
    """

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
        self.spd_r = int(spd_r)
        self.spd_eps = float(spd_eps)

        self.proj = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.spd_r),
        )
        depth = max(int(spd_mlp_depth), 1)
        layers: list[nn.Module] = [
            nn.Linear(self.spd_r, int(spd_latent_dim)),
            nn.GELU(),
            nn.Dropout(float(spd_dropout)),
            nn.LayerNorm(int(spd_latent_dim)),
        ]
        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.Linear(int(spd_latent_dim), int(spd_latent_dim)),
                    nn.GELU(),
                    nn.Dropout(float(spd_dropout)),
                    nn.LayerNorm(int(spd_latent_dim)),
                ]
            )
        layers.append(nn.Linear(int(spd_latent_dim), int(spd_latent_dim)))
        self.bottleneck = nn.Sequential(*layers)

        mu_t = None
        if pca_mu is not None:
            mu_t = torch.from_numpy(pca_mu).float() if not isinstance(pca_mu, torch.Tensor) else pca_mu.float()

        latent = int(spd_latent_dim)
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
        if x.dim() != 3:
            raise ValueError(f"Expected x shape (B,T,D), got {tuple(x.shape)}")
        z = self.proj(x)
        z = z.mean(dim=1)
        z = self.bottleneck(z)
        return self.linear_head(z)
