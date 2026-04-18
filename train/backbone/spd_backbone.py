import contextlib

import torch
import torch.nn as nn

from train.heads import SPEXHead, make_layernorm_linear_head
class SpdBackbone(nn.Module):
    """
    x(B,T,D) -> LayerNorm+Linear(D->r) -> covariance -> logm -> upper-tri vec
    -> bottleneck -> head
    """

    def __init__(
        self,
        *,
        input_dim: int,
        num_outputs: int,
        spd_r: int = 256,
        spd_eps: float = 1e-3,
        spd_latent_dim: int = 1024,
        spd_dropout: float = 0.1,
        spd_mlp_depth: int = 1,
        device: str | torch.device = "cuda:0",
        head_type: str = "linear",
        spex_k: int = 64,
        pca_mu=None,
        pca_U_k=None,
        spd_proj_mean=None,
        spd_proj_components=None,
        spd_desc_dim: int | None = None,
    ):
        super().__init__()
        self.device = device
        self.input_dim = int(input_dim)
        self.num_outputs = int(num_outputs)
        self.head_type = str(head_type).lower()
        self.r = int(spd_r)
        self.eps = float(spd_eps)

        _ = (spd_proj_mean, spd_proj_components, spd_desc_dim)

        self.proj = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.r),
        )
        tri_idx = torch.triu_indices(self.r, self.r)
        self.register_buffer("tri_i", tri_idx[0], persistent=False)
        self.register_buffer("tri_j", tri_idx[1], persistent=False)
        self.register_buffer("eye_r", torch.eye(self.r), persistent=False)

        vec_dim = self.r * (self.r + 1) // 2
        depth = max(int(spd_mlp_depth), 1)
        layers: list[nn.Module] = [
            nn.Linear(vec_dim, int(spd_latent_dim)),
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

        latent = int(spd_latent_dim)
        use_spex = self.head_type == "spex"
        mu_t = None
        if pca_mu is not None:
            mu_t = torch.from_numpy(pca_mu).float() if not isinstance(pca_mu, torch.Tensor) else pca_mu.float()

        if use_spex and pca_mu is not None and pca_U_k is not None:
            uk_t = torch.from_numpy(pca_U_k).float() if not isinstance(pca_U_k, torch.Tensor) else pca_U_k.float()
            self.linear_head = SPEXHead(latent, self.num_outputs, spex_k, mu_t, uk_t)
        else:
            base_head = make_layernorm_linear_head(latent, self.num_outputs)
            if use_spex:
                self.linear_head = SPEXHead(latent, self.num_outputs, spex_k)
            else:
                self.linear_head = base_head

    def _cov(self, H: torch.Tensor) -> torch.Tensor:
        B, T, d = H.shape
        Hc = H - H.mean(dim=1, keepdim=True)
        denom = float(max(T - 1, 1))
        C = torch.bmm(Hc.transpose(1, 2), Hc) / denom
        I = self.eye_r.to(device=H.device, dtype=H.dtype).unsqueeze(0).expand(B, -1, -1)
        tr = C.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        scale = (tr / float(d)).clamp_min(1e-6).view(B, 1, 1)
        C = C / scale
        alpha = 0.05
        C = (1.0 - alpha) * C + alpha * I
        return C + self.eps * I

    def _logm_spd(self, C: torch.Tensor) -> torch.Tensor:
        C = C.float()
        evals, evecs = torch.linalg.eigh(C)
        evals = torch.clamp(evals, min=self.eps)
        log_e = torch.log(evals)
        return evecs @ torch.diag_embed(log_e) @ evecs.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x shape (B,T,D), got {tuple(x.shape)}")
        ctx = torch.amp.autocast("cuda", enabled=False) if x.is_cuda else contextlib.nullcontext()
        with ctx:
            z = self.proj(x.float())
            C = self._cov(z)
            S = self._logm_spd(C)
            h = S[:, self.tri_i, self.tri_j]
            h = self.bottleneck(h)
            y = self.linear_head(h)
        return y
