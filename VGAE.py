import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import NNConv
from torch_geometric.utils import negative_sampling

class VGAEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, edge_attr_dim):
        super().__init__()
        # NNConv kernel nets map edge_attr -> weight matrix (in_dim x hidden_dim)
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        # kernel should produce weight matrices of shape (hidden_dim * hidden_dim)
        self.kernel1 = nn.Sequential(nn.Linear(edge_attr_dim, hidden_dim * hidden_dim), nn.ReLU())
        self.nnconv1 = NNConv(hidden_dim, hidden_dim, self.kernel1, aggr='mean')

        # final hidden to latent mu/logvar
        self.lin_mu = nn.Linear(hidden_dim, latent_dim)
        self.lin_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.lin_in(x))
        h = F.relu(self.nnconv1(h, edge_index, edge_attr))
        mu = self.lin_mu(h)
        logvar = self.lin_logvar(h)
        return mu, logvar


class EdgeDecoder(nn.Module):
    def __init__(self, z_dim, edge_attr_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim * 2 + edge_attr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z, edge_index, edge_attr):
        # edge_index: [2, E]
        src, dst = edge_index
        z_src = z[src]
        z_dst = z[dst]
        e = edge_attr if edge_attr is not None else torch.zeros((z_src.size(0), 0), device=z.device)
        inp = torch.cat([z_src, z_dst, e], dim=1)
        return self.mlp(inp).view(-1)