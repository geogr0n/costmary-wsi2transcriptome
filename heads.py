import torch
import torch.nn as nn


class SPEXHead(nn.Module):
    """z = W(h), y = mu + z @ U_k.T."""

    def __init__(
        self,
        base_dim: int,
        num_outputs: int,
        k: int,
        mu: torch.Tensor | None = None,
        U_k: torch.Tensor | None = None,
    ):
        super().__init__()
        self.base_dim = int(base_dim)
        self.num_outputs = int(num_outputs)
        self.k = int(k)
        self.z_proj = nn.Linear(base_dim, k)
        if mu is not None:
            self.register_buffer("mu", mu.float())
        else:
            self.register_buffer("mu", torch.zeros(num_outputs))

        basis = U_k.float() if U_k is not None else torch.zeros(self.num_outputs, self.k)
        self.register_buffer("U_k", basis)

    def set_pca(self, mu: torch.Tensor, U_k: torch.Tensor) -> None:
        self.register_buffer("mu", mu.to(self.mu.device if hasattr(self, "mu") else mu.device))
        target_basis = U_k.to(self.U_k.device if hasattr(self, "U_k") else U_k.device)
        if hasattr(self, "U_k"):
            self.U_k = target_basis
        else:
            self.register_buffer("U_k", target_basis)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.z_proj(h)
        return self.mu + z @ self.U_k.T


def make_layernorm_linear_head(hidden_dim: int, num_outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(int(hidden_dim)),
        nn.Linear(int(hidden_dim), int(num_outputs)),
    )
