"""
Loosely based on 10.48550/arXiv.2412.06264 (Flow Matching Guide and Code)

Also based on 10.48550/arXiv.2006.11239 (Denoising Diffusion Probabalistic Models)
    and 10.48550/arXiv.1706.03762 (Attention is All You Need)

"""

import torch
from torch import nn
from torchdiffeq import odeint
import math
from scvi.distributions import ZeroInflatedNegativeBinomial


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.mlp = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def get_sin_embeddings(self, t):
        half_dimension = self.embedding_dim // 2
        embedding_scale = math.log(10000) / (half_dimension - 1)
        embedding_sclae = torch.exp(torch.arange(half_dimension, device=t.device) * -embedding_scale)

        t = t.view(-1)
        embedding = t[:, None] * embedding_sclae[None, :]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)

        return embedding

    def forward(self, t):
        x = self.get_sin_embeddings(t)
        return self.mlp(x)


class FlowModel(nn.Module):
    def __init__(
        self,
        dim: int = 2,
        hidden_dim: int = 64,
        time_embedding_dim: int = 16,
        conditional_model: bool = False,
        num_perturbs: int | None = None,
        perturb_embedding_dim: int | None = None,
        num_conditions: int | None = None,
        condition_embedding_dim: int | None = None,
    ):
        super().__init__()
        self.conditional_model = conditional_model
        if self.conditional_model:
            assert num_conditions
            assert condition_embedding_dim
            assert perturb_embedding_dim
            self.condition_embedding = nn.Embedding(num_conditions, condition_embedding_dim)
            self.perturb_emb = nn.Embedding(num_perturbs, perturb_embedding_dim)
            total_cond_dims = perturb_embedding_dim + condition_embedding_dim
        else:
            total_cond_dims = 0

        self.time_output_dim = hidden_dim
        self.time_mlp = TimeEmbedding(embedding_dim=time_embedding_dim, hidden_dim=self.time_output_dim)

        self.fc1 = nn.Linear(dim + total_cond_dims + self.time_output_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + total_cond_dims, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim + total_cond_dims, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim + total_cond_dims, dim)
        self.act_func = nn.SiLU()

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, perturb_ids: torch.Tensor, condition_ids: torch.Tensor
    ) -> torch.Tensor:
        if self.conditional_model:
            c_emb = self.condition_embedding(condition_ids)
            p_emb = self.perturb_emb(perturb_ids)
            cond = torch.cat([p_emb, c_emb], dim=-1)
        else:
            cond = torch.empty(x.shape[0], 0, device=x.device)

        t_emb = self.time_mlp(t)

        h = torch.cat([x, t_emb, cond], dim=-1)
        h = self.fc1(h)
        h = self.act_func(self.ln1(h))
        h = torch.cat([h, cond], dim=-1)
        h = self.fc2(h)
        h = self.act_func(self.ln2(h))
        h = torch.cat([h, cond], dim=-1)
        h = self.fc3(h)
        h = self.act_func(self.ln3(h))
        h = torch.cat([h, cond], dim=-1)
        out = self.fc4(h)

        return out

    def integrate(
        self,
        x0: torch.Tensor,
        t0: float | torch.Tensor,
        t1: float | torch.Tensor,
        perturbations: torch.Tensor | int | None,
        conditions: torch.Tensor | int | None,
        **kwargs,
    ) -> torch.Tensor:
        ts = torch.tensor([t0, t1], device=x0.device)
        batch_size = x0.shape[0]

        if isinstance(perturbations, int):
            perturb_ids = torch.full((batch_size,), perturbations, device=x0.device)
        else:
            perturb_ids = perturbations.long().to(x0.device)

        if isinstance(conditions, int):
            cond_ids = torch.full((batch_size,), conditions, device=x0.device, dtype=torch.long)
        else:
            cond_ids = conditions.long().to(x0.device)

        def f(t, x):
            t_in = t.expand(x.shape[0], 1)
            return self.forward(x, t_in, perturb_ids, cond_ids)

        x_traj = odeint(f, x0, ts, **kwargs)
        return x_traj[-1]


class ZINBSampler(nn.Module):
    def __init__(self, ae_model: nn.Module, device: str = "cpu"):
        super().__init__()
        self.ae_model = ae_model
        self.device = device

    def sample(self, z: torch.Tensor, library_size: torch.Tensor) -> torch.Tensor:
        """
        decode latent representation z and samples from the ZINB dist
        """

        h_decoded = self.ae_model.decoder(z)
        mean_proportions = self.ae_model.decoder_mean(h_decoded)
        dispersion = torch.exp(self.ae_model.decoder_dispersion)
        dropout = self.ae_model.decoder_dropout(h_decoded)

        if library_size.dim() == 1:
            library_size = library_size.unsqueeze(1)

        dist = ZeroInflatedNegativeBinomial(mu=mean_proportions * library_size, theta=dispersion, zi_logits=dropout)
        return dist.sample()
