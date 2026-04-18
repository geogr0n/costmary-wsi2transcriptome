import os
import argparse
import pickle
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

from train.read_data import SuperTileRNADataset
from train.utils import patient_kfold, filter_no_features, custom_collate_fn
from train.backbone.tformer_lin import ViS
from train.backbone.he2rna_backbone import HE2RNABackbone
from train.backbone.spd_backbone import SpdBackbone
from train.backbone.mean_backbone import MeanBackbone
from train.backbone.projmean_backbone import ProjMeanBackbone
from train.training import train, evaluate

COSTMARY_ROOT = Path(__file__).resolve().parent.parent


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _resolve_repo_path(value):
    value = str(value or "").strip()
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = COSTMARY_ROOT / path
    return str(path.resolve())

def _param_counts(m: nn.Module):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


def _decoder_param_prefixes(m: nn.Module):
    prefixes = []
    if hasattr(m, "linear_head"):
        prefixes.append("linear_head.")
    if getattr(m, "classifier", None) is not None:
        prefixes.append("classifier.")
    return tuple(prefixes)


def _backbone_param_counts(m: nn.Module):
    exclude = _decoder_param_prefixes(m)
    total = 0
    trainable = 0
    for name, p in m.named_parameters():
        if exclude and any(name.startswith(prefix) for prefix in exclude):
            continue
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return total, trainable


def _backbone_param_count(m: nn.Module):
    return _backbone_param_counts(m)[1]

def _print_model_summary(model: nn.Module, *, model_type: str, decoder: str):
    backbone_params = _backbone_param_count(model)
    print(f"[model] type={model_type} class={model.__class__.__name__} decoder={decoder} "
          f"parameter_count={backbone_params:,}")

def _iter_dataset_features(dataset: SuperTileRNADataset):
    if getattr(dataset, "preloaded", False):
        for feat in dataset._features:
            yield feat
    else:
        for i in range(len(dataset)):
            feat, _, _, _ = dataset[i]
            yield feat


def _precompute_mean_inputs(dataset: SuperTileRNADataset):
    cached = []
    for feat in _iter_dataset_features(dataset):
        if feat is None:
            cached.append(None)
        else:
            cached.append(feat.mean(dim=0).detach().clone())
    dataset.set_precomputed_inputs(cached)

def _build_model(args, *, model_type: str, num_outputs: int, feature_dim: int, device: torch.device,
                 pca_mu=None, pca_U_k=None):
    head_type = str(getattr(args, "head_type")).lower()
    spex_k = int(pca_U_k.shape[1]) if pca_U_k is not None else getattr(args, "spex_k", 64)
    spd_r = int(getattr(args, "spd_r", 64))
    spd_latent = int(getattr(args, "spd_latent_dim", 256))

    if model_type == "vis":
        return ViS(
            num_outputs=num_outputs,
            input_dim=feature_dim,
            depth=args.depth,
            nheads=args.num_heads,
            dimensions_f=64,
            dimensions_c=64,
            dimensions_s=64,
            device=device,
            head_type=head_type,
            spex_k=spex_k,
            pca_mu=pca_mu,
            pca_U_k=pca_U_k,
        )
    if model_type == "spd":
        return SpdBackbone(
            input_dim=feature_dim,
            num_outputs=num_outputs,
            device=device,
            head_type=head_type,
            spex_k=spex_k,
            pca_mu=pca_mu,
            pca_U_k=pca_U_k,
            spd_r=spd_r,
            spd_eps=float(getattr(args, "spd_eps", 1e-3)),
            spd_latent_dim=1024,
            spd_mlp_depth=1,
            spd_dropout=0.1,
        )
    if model_type == "mean":
        return MeanBackbone(
            input_dim=feature_dim,
            num_outputs=num_outputs,
            device=device,
            head_type=head_type,
            spex_k=spex_k,
            pca_mu=pca_mu,
            pca_U_k=pca_U_k,
            spd_r=spd_r,
            spd_eps=float(getattr(args, "spd_eps", 1e-3)),
            spd_latent_dim=spd_latent,
            spd_dropout=getattr(args, "spd_dropout", 0.1),
            spd_mlp_depth=getattr(args, "spd_mlp_depth", 1),
        )
    if model_type == "projmean":
        return ProjMeanBackbone(
            input_dim=feature_dim,
            num_outputs=num_outputs,
            device=device,
            head_type=head_type,
            spex_k=spex_k,
            pca_mu=pca_mu,
            pca_U_k=pca_U_k,
            spd_r=spd_r,
            spd_eps=float(getattr(args, "spd_eps", 1e-3)),
            spd_latent_dim=spd_latent,
            spd_dropout=getattr(args, "spd_dropout", 0.1),
            spd_mlp_depth=getattr(args, "spd_mlp_depth", 1),
        )
    if model_type == "he2rna":
        he2rna_ks = tuple(int(x) for x in (args.he2rna_ks or "1,2,5,10,20,50,100").replace(" ", "").split(",") if x.strip())
        return HE2RNABackbone(
            input_dim=feature_dim,
            num_outputs=num_outputs,
            ks=he2rna_ks,
            dropout=args.he2rna_dropout,
            device=device,
            head_type=head_type,
            spex_k=spex_k,
            pca_mu=pca_mu,
            pca_U_k=pca_U_k,
        )
    raise ValueError(f"Unknown model_type: {model_type}")

def build_train_arg_parser():
    parser = argparse.ArgumentParser(description='Getting features')

    # general args
    parser.add_argument('--ref_file', type=str, default='', help='path to reference file')
    parser.add_argument('--feature_path', type=str, default='', help='path to resnet/uni and clustered features')
    parser.add_argument('--save_dir', type=str, default='', help='parent destination folder')
    parser.add_argument('--cohort', type=str, default="TCGA", help='cohort name for creating the saving folder of the results')
    parser.add_argument('--exp_name', type=str, default="exp", help='Experiment name for creating the saving folder of the results')
    parser.add_argument('--filter_no_features', type=int, default=1, help='Whether to filter out samples with no features')
    # model args
    parser.add_argument(
        '--model_type',
        type=str,
        default='vis',
        choices=['vis', 'he2rna', 'spd', 'mean', 'projmean'],
        help='Backbone type: vis | he2rna | spd | mean | projmean',
    )
    # SPD hyperparameters retained for spd / mean / projmean.
    parser.add_argument('--spd_r', type=int, default=64, help='SPD projection dim r')
    parser.add_argument('--spd_eps', type=float, default=1e-3, help='SPD eig/diagonal loading eps')
    parser.add_argument('--spd_latent_dim', type=int, default=1024, help='Shared latent dim for projmean (spd backbone uses fixed 1024)')
    parser.add_argument('--spd_mlp_depth', type=int, default=1, help='Shared stem depth for projmean (spd backbone uses fixed 1)')
    parser.add_argument('--spd_dropout', type=float, default=0.1, help='Shared dropout for projmean (spd backbone uses fixed 0.1)')
    parser.add_argument('--depth', type=int, default=6, help='transformer depth')
    parser.add_argument('--num-heads', type=int, default=16, help='number of attention heads')
    parser.add_argument('--he2rna_ks', type=str, default='1,2,5,10,20,50,100', help='HE2RNA top-k list')
    parser.add_argument('--he2rna_dropout', type=float, default=0.5, help='HE2RNA dropout')
    parser.add_argument('--head_type', type=str, default='linear',
                        choices=['linear', 'spex'],
                        help='Head: linear | spex')
    parser.add_argument('--spex_k', type=int, default=64, help='SPEX PCA components')
    parser.add_argument('--seed', type=int, default=29, help='Seed for random generation')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--train', help="if you want to train the model", action="store_true")
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--k', type=int, default=5, help='Number of splits')
    parser.add_argument('--save_on', type=str, default='loss+corr', help='which criterion to save model on: loss | corr | loss+corr')
    parser.add_argument('--stop_on', type=str, default='loss+corr', help='which criterion to stop on: loss | corr | loss+corr')

    # performance args（数据预加载到内存后 num_workers=0 即可）
    parser.add_argument('--num_workers_train', type=int, default=0, help='train dataloader num_workers')
    parser.add_argument('--num_workers_val', type=int, default=0, help='val dataloader num_workers')
    parser.add_argument('--num_workers_test', type=int, default=0, help='test dataloader num_workers')

    return parser


def parse_train_args(argv=None):
    """argv=None 时从 sys.argv 解析（供 `python -m train.main --ref_file ...`）；传入列表则用于测试。"""
    parser = build_train_arg_parser()
    args = parser.parse_args(argv)
    args.ref_file = _resolve_repo_path(args.ref_file)
    args.feature_path = _resolve_repo_path(args.feature_path)
    args.save_dir = _resolve_repo_path(args.save_dir)
    if not args.ref_file:
        parser.error("--ref_file is required")
    if not args.feature_path:
        parser.error("--feature_path is required")
    if not args.save_dir:
        parser.error("--save_dir is required")
    return args


if __name__ == '__main__':
    args = parse_train_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False # possibly reduced performance but better reproducibility
    torch.backends.cudnn.deterministic = True

    # reproducibility train dataloader
    g = torch.Generator()
    g.manual_seed(int(args.seed))

    save_dir = os.path.join(args.save_dir, args.cohort, args.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    run = None
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)

    # GPU 加速：启用 TF32
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    df = pd.read_csv(args.ref_file)

    if args.filter_no_features:
        df = filter_no_features(df, args.feature_path, 'cluster_features')

    train_idxs, val_idxs, test_idxs = patient_kfold(df, n_splits=args.k)

    test_results_splits = {}
    i = 0
    fold_train_seconds = []
    model_param_count = None

    for train_idx, val_idx, test_idx in zip(train_idxs, val_idxs, test_idxs):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        test_df = df.iloc[test_idx]

        # save patient ids to file
        np.save(save_dir + '/train_'+str(i)+'.npy', np.unique(train_df.patient_id) )
        np.save(save_dir + '/val_'+str(i)+'.npy', np.unique(val_df.patient_id) )
        np.save(save_dir + '/test_'+str(i)+'.npy', np.unique(test_df.patient_id) )
    
        # init dataset
        train_dataset = SuperTileRNADataset(train_df, args.feature_path)
        val_dataset = SuperTileRNADataset(val_df, args.feature_path)
        test_dataset = SuperTileRNADataset(test_df, args.feature_path)

        num_outputs = train_dataset.num_genes
        feature_dim = train_dataset.feature_dim

        pca_mu, pca_U_k = None, None
        head_type = str(getattr(args, "head_type")).lower()
        if head_type == "spex":
            rna_cols = [c for c in train_df.columns if c.startswith("rna_")]
            Y_train = train_df[rna_cols].values.astype(np.float32)
            k = min(getattr(args, "spex_k", 64), Y_train.shape[1], max(1, Y_train.shape[0] - 1))
            from sklearn.decomposition import PCA
            pca = PCA(n_components=k)
            pca.fit(Y_train)
            pca_mu = pca.mean_.astype(np.float32)
            pca_U_k = pca.components_.T.astype(np.float32)
        if args.model_type == "mean":
            _precompute_mean_inputs(train_dataset)
            _precompute_mean_inputs(val_dataset)
            _precompute_mean_inputs(test_dataset)

        # init dataloaders
        def _dl_kwargs(num_workers, shuffle):
            kw = dict(
                num_workers=int(num_workers),
                pin_memory=True,
                shuffle=shuffle,
                batch_size=args.batch_size,
                collate_fn=custom_collate_fn,
            )
            if int(num_workers) > 0:
                kw.update(persistent_workers=True, prefetch_factor=2)
            return kw

        train_dataloader = DataLoader(train_dataset, 
                    worker_init_fn=seed_worker if args.num_workers_train > 0 else None,
                    generator=g,
                    **_dl_kwargs(args.num_workers_train, shuffle=True))
    
        val_dataloader = DataLoader(val_dataset, 
                    **_dl_kwargs(args.num_workers_val, shuffle=False))
    
        test_dataloader = DataLoader(test_dataset, 
                    **_dl_kwargs(args.num_workers_test, shuffle=False))
    
        model = _build_model(
            args,
            model_type=args.model_type,
            num_outputs=num_outputs,
            feature_dim=feature_dim,
            device=device,
            pca_mu=pca_mu,
            pca_U_k=pca_U_k,
        )

        model.to(device)

        # training 
        dataloaders = { 'train': train_dataloader, 'val': val_dataloader}
        _print_model_summary(
            model,
            model_type=args.model_type,
            decoder="baseline",
        )
        if model_param_count is None:
            model_param_count = _backbone_param_count(model)
        train_lr = args.lr
        train_weight_decay = 0.0
        train_epochs = args.num_epochs
        train_patience = int(args.patience)
        if args.model_type in ('he2rna', 'vis'):
            train_lr = args.lr
            train_weight_decay = 0.0
            train_epochs = args.num_epochs
            train_patience = int(args.patience)
            optimizer = torch.optim.Adam(list(model.parameters()), lr=train_lr, weight_decay=train_weight_decay)
        else:
            optimizer = torch.optim.AdamW(list(model.parameters()),
                                            lr=train_lr, amsgrad=False, weight_decay=train_weight_decay)

        if args.train:
            train_start = time.perf_counter()
            model = train(model, dataloaders, optimizer,
                            num_epochs=train_epochs, run=run, split=i, 
                            save_on=args.save_on, stop_on=args.stop_on, 
                            delta=0.5, save_dir=save_dir, patience=train_patience)
            fold_train_seconds.append(float(time.perf_counter() - train_start))

        preds, real, wsis, projs = evaluate(model, test_dataloader, run=run, suff='_'+str(i))

        # random model: same backbone + random weights. For SPEX, use random mu and random orthogonal U_ran so baseline is not inflated by structure prior.
        if head_type == "spex" and pca_mu is not None and pca_U_k is not None:
            k = pca_U_k.shape[1]
            rng = np.random.default_rng(int(args.seed) + 1000 + i)
            mu_scale = np.std(pca_mu) + 1e-8
            mu_loc = np.mean(pca_mu)
            random_pca_mu = (rng.standard_normal(num_outputs) * mu_scale + mu_loc).astype(np.float32)
            A = rng.standard_normal((num_outputs, k))
            Q, _ = np.linalg.qr(A)
            random_pca_U = Q.astype(np.float32)
        else:
            random_pca_mu = pca_mu
            random_pca_U = pca_U_k

        def _make_random_model():
            return _build_model(
                args,
                model_type=args.model_type,
                num_outputs=num_outputs,
                feature_dim=feature_dim,
                device=device,
                pca_mu=random_pca_mu,
                pca_U_k=random_pca_U,
            )
        random_model = _make_random_model()
        random_model = random_model.to(device)
        random_preds, _, _, _ = evaluate(random_model, test_dataloader, run=run, suff='_'+str(i)+'_rand')
    
        test_results = {
            'real': real,
            'preds': preds,
            'random': random_preds,
            'wsi_file_name': wsis,
            'tcga_project': projs
        }
    
        test_results_splits[f'split_{i}'] = test_results
        i += 1

    test_results_splits['genes'] = [x[4:] for x in df.columns if 'rna_' in x]
    with open(os.path.join(save_dir, 'test_results.pkl'), 'wb') as f:
        pickle.dump(test_results_splits, f, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        'experiment': args.exp_name,
        'model_type': args.model_type,
        'head_type': str(getattr(args, 'head_type', 'linear')).lower(),
        'spex_k': int(getattr(args, 'spex_k', 64)),
        'num_folds': int(i),
        'parameter_count': int(model_param_count or 0),
        'fold_train_seconds': [float(x) for x in fold_train_seconds],
        'avg_fold_train_seconds': float(np.mean(fold_train_seconds)) if fold_train_seconds else None,
        'sum_fold_train_seconds': float(np.sum(fold_train_seconds)) if fold_train_seconds else None,
    }
    with open(os.path.join(save_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
