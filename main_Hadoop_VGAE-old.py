"""
VGAE training script for Hadoop dataset (directed graphs with node and edge attributes).
Encoder: NNConv-based variational encoder using edge_attr.
Decoder: edge-conditioned MLP on [z_u || z_v || edge_attr].

Quick smoke test: runs a few epochs and prints losses.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace
from sklearn.metrics import average_precision_score, roc_auc_score
import pickle

from torch_geometric.nn import NNConv
from torch_geometric.utils import negative_sampling

from DataLoader import create_loaders


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


def reparametrize(mu, logvar):
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std


def train_epoch(model_enc, model_dec, optimizer, loader, device, kl_weight=1e-3):
    model_enc.train()
    model_dec.train()
    total_loss = 0.0
    bcount = 0

    for batch in loader:
        batch = batch.to(device)
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        mu, logvar = model_enc(x, edge_index, edge_attr)
        z = reparametrize(mu, logvar)

        # positive edges (existing)
        pos_edge_index = edge_index
        pos_logits = model_dec(z, pos_edge_index, edge_attr)

        # negative edges: sample per graph to avoid cross-graph negatives
        neg_edge_index = []
        neg_edge_attr = []
        # iterate graphs in batch
        for g in range(batch.num_graphs):
            node_mask = (batch.batch == g).nonzero(as_tuple=False).view(-1)
            if node_mask.numel() < 2:
                continue
            # get local mapping
            # edges within this graph
            mask_src = (edge_index[0] >= node_mask[0]) & (edge_index[0] <= node_mask[-1])
            mask_dst = (edge_index[1] >= node_mask[0]) & (edge_index[1] <= node_mask[-1])
            mask_both = mask_src & mask_dst
            pos_idx = mask_both.nonzero(as_tuple=False).view(-1)
            num_pos = pos_idx.size(0) if pos_idx is not None else 0
            if num_pos == 0:
                continue
            n_nodes = node_mask.numel()
            # localize pos edges to [0, n_nodes-1] for negative_sampling
            local_pos = edge_index[:, pos_idx] - node_mask[0]
            neg_local = negative_sampling(edge_index=local_pos, num_nodes=n_nodes, num_neg_samples=num_pos)
            # remap to global ids
            neg_global = neg_local.clone()
            neg_global[0] = node_mask[neg_global[0]]
            neg_global[1] = node_mask[neg_global[1]]
            neg_edge_index.append(neg_global)
            # use zeros for edge_attr of negative edges
            if edge_attr is not None:
                neg_edge_attr.append(torch.zeros((neg_global.size(1), edge_attr.size(1)), device=edge_attr.device))

        if len(neg_edge_index) > 0:
            neg_edge_index = torch.cat(neg_edge_index, dim=1)
            neg_edge_attr = torch.cat(neg_edge_attr, dim=0) if len(neg_edge_attr) > 0 else None
        else:
            neg_edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
            neg_edge_attr = torch.empty((0, edge_attr.size(1) if edge_attr is not None else 0), device=edge_attr.device)

        neg_logits = model_dec(z, neg_edge_index, neg_edge_attr)

        # labels
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0).to(logits.device)

        recon_loss = F.binary_cross_entropy_with_logits(logits, labels)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon_loss + kl_weight * kld

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        bcount += 1

    return total_loss / max(1, bcount)


def test_epoch(model_enc, model_dec, loader, device, kl_weight=1e-3):
    model_enc.eval()
    model_dec.eval()

    scores_all = []
    labels_all = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

            mu, logvar = model_enc(x, edge_index, edge_attr)
            z = mu  # use mean for evaluation

            pos_edge_index = edge_index
            pos_logits = model_dec(z, pos_edge_index, edge_attr)

            # sample negatives per graph (same logic as train)
            neg_edge_index = []
            neg_edge_attr = []
            for g in range(batch.num_graphs):
                node_mask = (batch.batch == g).nonzero(as_tuple=False).view(-1)
                if node_mask.numel() < 2:
                    continue
                mask_src = (edge_index[0] >= node_mask[0]) & (edge_index[0] <= node_mask[-1])
                mask_dst = (edge_index[1] >= node_mask[0]) & (edge_index[1] <= node_mask[-1])
                mask_both = mask_src & mask_dst
                pos_idx = mask_both.nonzero(as_tuple=False).view(-1)
                num_pos = pos_idx.size(0) if pos_idx is not None else 0
                if num_pos == 0:
                    continue
                n_nodes = node_mask.numel()
                local_pos = edge_index[:, pos_idx] - node_mask[0]
                neg_local = negative_sampling(edge_index=local_pos, num_nodes=n_nodes, num_neg_samples=num_pos)
                neg_global = neg_local.clone()
                neg_global[0] = node_mask[neg_global[0]]
                neg_global[1] = node_mask[neg_global[1]]
                neg_edge_index.append(neg_global)
                if edge_attr is not None:
                    neg_edge_attr.append(torch.zeros((neg_global.size(1), edge_attr.size(1)), device=edge_attr.device))

            if len(neg_edge_index) > 0:
                neg_edge_index = torch.cat(neg_edge_index, dim=1)
                neg_edge_attr = torch.cat(neg_edge_attr, dim=0) if len(neg_edge_attr) > 0 else None
            else:
                neg_edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
                neg_edge_attr = torch.empty((0, edge_attr.size(1) if edge_attr is not None else 0), device=edge_attr.device)

            neg_logits = model_dec(z, neg_edge_index, neg_edge_attr)

            # per-edge losses
            pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits), reduction='none')
            neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits), reduction='none')

            # map edges to graphs
            if pos_edge_index.numel() > 0:
                pos_graph = batch.batch[pos_edge_index[0]]
            else:
                pos_graph = torch.tensor([], dtype=torch.long, device=device)
            if neg_edge_index.numel() > 0:
                neg_graph = batch.batch[neg_edge_index[0]]
            else:
                neg_graph = torch.tensor([], dtype=torch.long, device=device)

            all_losses = torch.cat([pos_loss, neg_loss], dim=0)
            all_graph = torch.cat([pos_graph, neg_graph], dim=0)

            # compute graph-level score as mean edge loss per graph
            for g in range(batch.num_graphs):
                mask = (all_graph == g)
                if mask.sum() == 0:
                    # fallback: use 0
                    scores_all.append(0.0)
                else:
                    scores_all.append(all_losses[mask].mean().item())

            labels_all.append(batch.y.cpu())

    labels_all = torch.cat(labels_all).cpu().numpy()
    scores_all = np.array(scores_all)

    # compute AP and ROC-AUC (if both classes present)
    try:
        ap = average_precision_score(y_true=labels_all, y_score=scores_all)
    except Exception:
        ap = -1
    try:
        roc = roc_auc_score(y_true=labels_all, y_score=scores_all)
    except Exception:
        roc = -1

    return ap, roc, scores_all, labels_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='Hadoop')
    parser.add_argument('--data_seed', type=int, default=1213)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=2000)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--save_output', action='store_true', help='Save epoch logs to outputs/*.pkl')
    args = parser.parse_args()

    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader, num_features, train_dataset, test_dataset, raw_dataset = create_loaders(data_name=args.data, batch_size=args.batch, dense=False, data_seed=args.data_seed)

    # get edge_attr dim from raw_dataset
    edge_attr_dim = raw_dataset.num_edge_attributes if hasattr(raw_dataset, 'num_edge_attributes') else 0

    enc = VGAEEncoder(in_dim=num_features, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim, edge_attr_dim=edge_attr_dim)
    dec = EdgeDecoder(z_dim=args.latent_dim, edge_attr_dim=edge_attr_dim, hidden_dim=args.hidden_dim)

    enc = enc.to(device)
    dec = dec.to(device)

    params = list(enc.parameters()) + list(dec.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    epochinfo = []

    for epoch in range(1, args.epochs + 1):
        print("\n+++++++++++++++++++VGAE++++++++++++++++++++++")
        print("Epoch %3d" % (epoch), end="\t")

        train_loss = train_epoch(enc, dec, optimizer, train_loader, device)
        print("Train loss: %f" % (train_loss), end="\t")

        ap, roc, scores, labels = test_epoch(enc, dec, test_loader, device)
        print("AP: %f" % (ap), end="\t")
        print("ROC-AUC: %f" % (roc))

        TEMP = SimpleNamespace()
        TEMP.epoch_no = epoch
        TEMP.train_loss = train_loss
        TEMP.ap = ap
        TEMP.roc_auc = roc
        TEMP.scores = scores
        TEMP.labels = labels
        epochinfo.append(TEMP)

    # choose best epoch by max ROC-AUC
    roc_list = [e.roc_auc for e in epochinfo]
    best_idx = int(np.nanargmax(roc_list)) if len(roc_list) > 0 else 0
    print("      Best ROC-AUC at epoch %d: ROC-AUC: %.3f" % (epochinfo[best_idx].epoch_no, epochinfo[best_idx].roc_auc))

    # save outputs similar to main_Hadoop.py
    if args.save_output:
        outdir = 'outputs'
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        serial = {
            'args': vars(args),
            'epochinfo': []
        }
        for e in epochinfo:
            serial['epochinfo'].append({
                'epoch_no': int(e.epoch_no),
                'train_loss': float(e.train_loss),
                'ap': float(e.ap) if e.ap is not None else None,
                'roc_auc': float(e.roc_auc) if e.roc_auc is not None else None,
                'scores': e.scores.tolist() if hasattr(e, 'scores') and e.scores is not None else None,
                'labels': e.labels.tolist() if hasattr(e, 'labels') and e.labels is not None else None,
            })

        outpath = os.path.join(outdir, f'VGAE_{args.data}_{args.data_seed}.pkl')
        with open(outpath, 'wb') as f:
            pickle.dump(serial, f)

        print('Saved outputs to', outpath)

    print('Quick training finished.')


if __name__ == '__main__':
    main()
