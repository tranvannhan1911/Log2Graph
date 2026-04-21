"""
VGAE training script for Hadoop dataset (directed graphs with node and edge attributes).
"""

import os
import sys
import argparse
import torch
import numpy as np
from types import SimpleNamespace
from sklearn.metrics import average_precision_score, roc_auc_score
import pickle
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

from DataLoader import create_loaders, VGAETrainer, build_fixed_negative_edges
from VGAE import VGAEEncoder, EdgeDecoder

# 🔥 Tee logger: print ra cả console + file
class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='Hadoop')
    parser.add_argument('--data_seed', type=int, default=1213)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=2000)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--save_output', action='store_true')
    parser.add_argument('--stopearly', type=int, default=50)
    args = parser.parse_args()

    # 🔥 Tạo thư mục history
    log_dir = "history"
    os.makedirs(log_dir, exist_ok=True)

    # 🔥 Tạo filename theo datetime
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"main_{args.data}_VGAE-{now}.log")

    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    print(f"Logging to: {log_path}")

    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    train_loader, test_loader, num_features, train_dataset, test_dataset, raw_dataset = create_loaders(
        data_name=args.data,
        batch_size=args.batch,
        dense=False,
        data_seed=args.data_seed
    )

    edge_attr_dim = raw_dataset.num_edge_attributes if hasattr(raw_dataset, 'num_edge_attributes') else 0

    enc = VGAEEncoder(
        in_dim=num_features,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        edge_attr_dim=edge_attr_dim
    ).to(device)

    dec = EdgeDecoder(
        z_dim=args.latent_dim,
        edge_attr_dim=edge_attr_dim,
        hidden_dim=args.hidden_dim
    ).to(device)

    params = list(enc.parameters()) + list(dec.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10
    )

    trainer = VGAETrainer(enc, dec, optimizer, device=device)

    print("Building fixed negative edges for test set...")
    test_neg_edges = build_fixed_negative_edges(test_loader, device)
    trainer.set_test_negatives(test_neg_edges)

    epochinfo = []
    
    best_roc = -1
    patience_cnt = 0

    for epoch in range(1, args.epochs + 1):
        print("\n+++++++++++++++++++VGAE++++++++++++++++++++++")
        print("Epoch %3d" % epoch, end="\t")

        train_loss = trainer.train(train_loader)
        print("Train loss: %f" % train_loss, end="\t")

        ap, roc, scores, labels = trainer.test(test_loader)
        print("AP: %f" % ap, end="\t")
        print("ROC-AUC: %f" % roc)

        scheduler.step(roc)

        TEMP = SimpleNamespace()
        TEMP.epoch_no = epoch
        TEMP.train_loss = train_loss
        TEMP.ap = ap
        TEMP.roc_auc = roc
        TEMP.scores = scores
        TEMP.labels = labels
        epochinfo.append(TEMP)

        if roc > best_roc:
            best_roc = roc
            patience_cnt = 0
        else:
            patience_cnt += 1
            if args.stopearly > 0 and patience_cnt >= args.stopearly:
                print(f"Early stopping at epoch {epoch} (patience={args.stopearly})")
                break


    # best epoch
    roc_list = [e.roc_auc for e in epochinfo]
    best_idx = int(np.nanargmax(roc_list)) if len(roc_list) > 0 else 0

    print("Best ROC-AUC at epoch %d: AP: %.3f\tROC-AUC: %.3f" % (
        epochinfo[best_idx].epoch_no,
        epochinfo[best_idx].ap,
        epochinfo[best_idx].roc_auc
    ))

    # save output
    if args.save_output:
        outdir = 'outputs'
        os.makedirs(outdir, exist_ok=True)

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
                'scores': e.scores.tolist() if e.scores is not None else None,
                'labels': e.labels.tolist() if e.labels is not None else None,
            })

        outpath = os.path.join(outdir, f'VGAE_{args.data}_{args.data_seed}.pkl')
        with open(outpath, 'wb') as f:
            pickle.dump(serial, f)

        print('Saved outputs to', outpath)

    print('Quick training finished.')

    log_file.close()


if __name__ == '__main__':
    main()