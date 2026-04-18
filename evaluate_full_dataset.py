import argparse
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from train.main import (
    _backbone_param_count,
    _build_model,
    _precompute_mean_inputs,
    _resolve_repo_path,
    seed_worker,
)
from train.read_data import SuperTileRNADataset
from train.training import evaluate
from train.utils import custom_collate_fn, filter_no_features


EXPERIMENT_DEFAULTS = {
    "mean": {"model_type": "mean", "head_type": "linear"},
    "mean_spex": {"model_type": "mean", "head_type": "spex", "spex_k": 16},
    "he2rna": {"model_type": "he2rna", "head_type": "linear"},
    "he2rna_spex": {"model_type": "he2rna", "head_type": "spex", "spex_k": 16},
    "vis": {"model_type": "vis", "head_type": "linear", "depth": 6, "num_heads": 16},
    "vis_spex": {"model_type": "vis", "head_type": "spex", "spex_k": 16, "depth": 6, "num_heads": 16},
    "spd": {"model_type": "spd", "head_type": "linear"},
    "spd_spex": {"model_type": "spd", "head_type": "spex", "spex_k": 16},
    # Legacy aliases retained for backward compatibility with older result trees.
    "mean_linear": {"model_type": "mean", "head_type": "linear"},
    "mean_spex_k16": {"model_type": "mean", "head_type": "spex", "spex_k": 16},
    "he2rna_linear": {"model_type": "he2rna", "head_type": "linear"},
    "he2rna_spex_k16": {"model_type": "he2rna", "head_type": "spex", "spex_k": 16},
    "vis_linear": {"model_type": "vis", "head_type": "linear", "depth": 6, "num_heads": 16},
    "vis_spex_k16": {"model_type": "vis", "head_type": "spex", "spex_k": 16, "depth": 6, "num_heads": 16},
    "spd_linear": {"model_type": "spd", "head_type": "linear"},
    "spd_spex_k16": {"model_type": "spd", "head_type": "spex", "spex_k": 16},
    "projmean": {"model_type": "projmean", "head_type": "linear"},
    "vis_d2": {"model_type": "vis", "head_type": "linear", "depth": 2, "num_heads": 16},
}


def rz_ci(r, n, conf_level=0.95):
    from math import atanh, pow
    from numpy import tanh
    from scipy.stats import norm

    zr_se = pow(1 / (n - 3), 0.5)
    moe = norm.ppf(1 - (1 - conf_level) / float(2)) * zr_se
    zu = atanh(r) + moe
    zl = atanh(r) - moe
    return tanh((zl, zu))


def rho_rxy_rxz(rxy, rxz, ryz):
    from math import pow

    num = (ryz - 1 / 2.0 * rxy * rxz) * (1 - pow(rxy, 2) - pow(rxz, 2) - pow(ryz, 2)) + pow(ryz, 3)
    den = (1 - pow(rxy, 2)) * (1 - pow(rxz, 2))
    return num / float(den)


def dependent_corr(xy, xz, yz, n, twotailed=True, conf_level=0.95, method="steiger"):
    if method == "steiger":
        from scipy.stats import t

        d = xy - xz
        determin = 1 - xy * xy - xz * xz - yz * yz + 2 * xy * xz * yz
        av = (xy + xz) / 2
        cube = (1 - yz) * (1 - yz) * (1 - yz)

        t2 = d * np.sqrt((n - 1) * (1 + yz) / (((2 * (n - 1) / (n - 3)) * determin + av * av * cube)))
        p = 1 - t.cdf(abs(t2), n - 3)
        if twotailed:
            p *= 2
        return t2, p
    if method == "zou":
        from math import pow

        l1 = rz_ci(xy, n, conf_level=conf_level)[0]
        u1 = rz_ci(xy, n, conf_level=conf_level)[1]
        l2 = rz_ci(xz, n, conf_level=conf_level)[0]
        u2 = rz_ci(xz, n, conf_level=conf_level)[1]
        rho_r12_r13 = rho_rxy_rxz(xy, xz, yz)
        lower = xy - xz - pow((pow((xy - l1), 2) + pow((u2 - xz), 2) - 2 * rho_r12_r13 * (xy - l1) * (u2 - xz)), 0.5)
        upper = xy - xz + pow((pow((u1 - xy), 2) + pow((xz - l2), 2) - 2 * rho_r12_r13 * (u1 - xy) * (xz - l2)), 0.5)
        return lower, upper
    raise Exception("Wrong method!")


def fdrcorrection_bh(pvals, alpha=0.05):
    pvals = np.asarray(pvals, dtype=float)
    if pvals.size == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    order = np.argsort(pvals)
    sorted_p = pvals[order]
    n = len(sorted_p)
    ecdf = np.arange(1, n + 1, dtype=float) / float(n)

    reject_sorted = np.zeros(n, dtype=bool)
    passed = sorted_p <= alpha * ecdf
    if np.any(passed):
        last = np.max(np.where(passed)[0])
        reject_sorted[: last + 1] = True

    adjusted_sorted = sorted_p / ecdf
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    reject = np.empty_like(reject_sorted)
    adjusted = np.empty_like(adjusted_sorted)
    reject[order] = reject_sorted
    adjusted[order] = adjusted_sorted
    return reject, adjusted


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Pure test-only evaluation for released costmary checkpoints on the main cohort or any external cohort prepared under the same input contract."
    )
    parser.add_argument("--ref_file", type=str, required=True, help="Path to the prepared reference CSV.")
    parser.add_argument("--feature_path", type=str, required=True, help="Path to clustered slide features.")
    parser.add_argument("--save_dir", type=str, required=True, help="Parent directory for exported evaluation outputs.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to a downloaded model checkpoint.")
    parser.add_argument("--cohort", type=str, default="TCGA", help="Cohort label used for organizing outputs.")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name for the evaluated checkpoint.")
    parser.add_argument("--metadata_json", type=str, default="", help="Optional training metadata JSON to merge into outputs.")
    parser.add_argument("--filter_no_features", type=int, default=1, help="Whether to filter samples with missing feature files.")
    parser.add_argument("--model_type", type=str, default="", choices=["", "vis", "he2rna", "spd", "mean", "projmean"],
                        help="Backbone type. If omitted, inferred from --exp_name for the standard released checkpoints.")
    parser.add_argument("--head_type", type=str, default="", choices=["", "linear", "spex"],
                        help="Decoder type. If omitted, inferred from --exp_name when possible.")
    parser.add_argument("--spex_k", type=int, default=0, help="SPEX rank. If omitted, inferred from --exp_name when possible.")
    parser.add_argument("--depth", type=int, default=6, help="Transformer depth for the ViS/SEQUOIA route.")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads for the ViS/SEQUOIA route.")
    parser.add_argument("--he2rna_ks", type=str, default="1,2,5,10,20,50,100", help="HE2RNA top-k list.")
    parser.add_argument("--he2rna_dropout", type=float, default=0.5, help="HE2RNA dropout.")
    parser.add_argument("--spd_r", type=int, default=64, help="SPD projection dimension.")
    parser.add_argument("--spd_eps", type=float, default=1e-3, help="SPD stabilization epsilon.")
    parser.add_argument("--spd_latent_dim", type=int, default=1024, help="Latent dimension for projmean.")
    parser.add_argument("--spd_mlp_depth", type=int, default=1, help="MLP depth for projmean.")
    parser.add_argument("--spd_dropout", type=float, default=0.1, help="Dropout for projmean.")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--num_workers_test", type=int, default=0, help="Number of dataloader workers.")
    parser.add_argument("--seed", type=int, default=29, help="Random seed for the random baseline.")
    return parser


def parse_args(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.ref_file = _resolve_repo_path(args.ref_file)
    args.feature_path = _resolve_repo_path(args.feature_path)
    args.save_dir = _resolve_repo_path(args.save_dir)
    args.checkpoint_path = _resolve_repo_path(args.checkpoint_path)
    args.metadata_json = _resolve_repo_path(args.metadata_json)
    return args


def load_optional_metadata(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_experiment_defaults(args, source_metadata):
    defaults = EXPERIMENT_DEFAULTS.get(args.exp_name, {})

    if not args.model_type:
        args.model_type = source_metadata.get("model_type") or defaults.get("model_type", "")
    if not args.head_type:
        args.head_type = source_metadata.get("head_type") or defaults.get("head_type", "")
    if not args.spex_k:
        args.spex_k = int(source_metadata.get("spex_k") or defaults.get("spex_k", 0) or 0)

    if "depth" in defaults and args.depth == 6:
        args.depth = defaults["depth"]
    if "num_heads" in defaults and args.num_heads == 16:
        args.num_heads = defaults["num_heads"]

    if not args.model_type:
        raise ValueError("Could not infer --model_type. Please provide it explicitly.")
    if not args.head_type:
        raise ValueError("Could not infer --head_type. Please provide it explicitly.")
    if args.head_type == "spex" and int(args.spex_k) <= 0:
        raise ValueError("SPEX checkpoints require --spex_k.")

    return args


def make_dataloader(dataset, batch_size, num_workers, seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=int(num_workers),
        pin_memory=True,
    )
    if int(num_workers) > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2, worker_init_fn=seed_worker, generator=generator)
    return DataLoader(dataset, **kwargs)


def make_random_spex_basis(num_outputs, spex_k, seed, label_matrix=None):
    rng = np.random.default_rng(int(seed))
    if label_matrix is not None:
        mu_scale = float(np.std(label_matrix.mean(axis=0))) + 1e-8
        mu_loc = float(np.mean(label_matrix.mean(axis=0)))
    else:
        mu_scale = 1.0
        mu_loc = 0.0
    random_pca_mu = (rng.standard_normal(num_outputs) * mu_scale + mu_loc).astype(np.float32)
    basis_seed = rng.standard_normal((num_outputs, int(spex_k)))
    q, _ = np.linalg.qr(basis_seed)
    return random_pca_mu, q.astype(np.float32)


def evaluate_predictions(preds, labels, random_preds, genes, experiment, metadata):
    pred_r, random_r, test_p, pearson_p = [], [], [], []
    rmse_pred, rmse_random, rmse_quantile_norm, rmse_mean_norm = [], [], [], []

    for j, gene in enumerate(genes):
        r = labels[:, j]
        p = preds[:, j]
        rand = random_preds[:, j]

        if len(set(p.tolist())) == 1 or len(set(r.tolist())) == 1 or len(set(rand.tolist())) == 1:
            xy, xz, yz, pv, p1 = 0.0, 0.0, 0.0, 1.0, 1.0
        else:
            xy, p1 = stats.pearsonr(r, p)
            xz, _ = stats.pearsonr(r, rand)
            yz, _ = stats.pearsonr(p, rand)
            _, pv = dependent_corr(xy, xz, yz, len(r), twotailed=False, conf_level=0.95, method="steiger")

        pred_r.append(float(xy))
        random_r.append(float(xz))
        test_p.append(float(pv))
        pearson_p.append(float(p1))

        rmse_p = float(np.sqrt(mean_squared_error(r, p)))
        rmse_r = float(np.sqrt(mean_squared_error(r, rand)))
        rmse_q = rmse_p / (float(np.quantile(r, 0.75) - np.quantile(r, 0.25)) + 1e-5)
        rmse_m = rmse_p / (float(np.mean(r)) + 1e-8)

        rmse_pred.append(rmse_p)
        rmse_random.append(rmse_r)
        rmse_quantile_norm.append(rmse_q)
        rmse_mean_norm.append(rmse_m)

    combine_res = pd.DataFrame(
        {
            "pred_real_r": pred_r,
            "random_real_r": random_r,
            "pearson_p": pearson_p,
            "Steiger_p": test_p,
            "rmse_pred": rmse_pred,
            "rmse_random": rmse_random,
            "rmse_quantile_norm": rmse_quantile_norm,
            "rmse_mean_norm": rmse_mean_norm,
        },
        index=genes,
    )

    combine_res = combine_res.sort_values("pred_real_r", ascending=False)
    combine_res["pred_real_r"] = combine_res["pred_real_r"].fillna(0)
    combine_res["random_real_r"] = combine_res["random_real_r"].fillna(0)

    combine_res["pearson_p"] = combine_res["pearson_p"].fillna(1)
    _, fdr_pearson_p = fdrcorrection_bh(combine_res["pearson_p"].values)
    combine_res["fdr_pearson_p"] = fdr_pearson_p

    combine_res["Steiger_p"] = combine_res["Steiger_p"].fillna(1)
    _, fdr_steiger_p = fdrcorrection_bh(combine_res["Steiger_p"].values)
    combine_res["fdr_Steiger_p"] = fdr_steiger_p

    combine_res["cancer"] = experiment

    col = combine_res["rmse_quantile_norm"]
    mn, mx = col.min(), col.max()
    combine_res["nrmse_01"] = (col - mn) / (mx - mn + 1e-8)

    sig_res = combine_res[
        (combine_res["pred_real_r"] > 0)
        & (combine_res["pearson_p"] < 0.05)
        & (combine_res["rmse_pred"] < combine_res["rmse_random"])
        & (combine_res["pred_real_r"] > combine_res["random_real_r"])
        & (combine_res["Steiger_p"] < 0.05)
        & (combine_res["fdr_Steiger_p"] < 0.2)
    ]

    top1000 = combine_res.head(1000)
    summary = {
        "experiment": experiment,
        "num_sig_genes": int(len(sig_res)),
        "top1000_mean_pcc": float(top1000["pred_real_r"].mean()),
        "all_mean_pcc": float(combine_res["pred_real_r"].mean()),
        "top1000_mean_rmse": float(top1000["rmse_pred"].mean()),
        "all_mean_rmse": float(combine_res["rmse_pred"].mean()),
        "top1000_mean_nrmse": float(top1000["nrmse_01"].mean()),
        "all_mean_nrmse": float(combine_res["nrmse_01"].mean()),
        "avg_fold_train_seconds": metadata.get("avg_fold_train_seconds"),
        "sum_fold_train_seconds": metadata.get("sum_fold_train_seconds"),
        "parameter_count": metadata.get("parameter_count"),
        "model_type": metadata.get("model_type"),
        "head_type": metadata.get("head_type"),
    }
    return combine_res, sig_res, summary


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    source_metadata = load_optional_metadata(args.metadata_json)
    args = apply_experiment_defaults(args, source_metadata)

    save_dir = os.path.join(args.save_dir, args.cohort, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    df = pd.read_csv(args.ref_file)
    if args.filter_no_features:
        df = filter_no_features(df, args.feature_path, "cluster_features")

    dataset = SuperTileRNADataset(df, args.feature_path)
    if args.model_type == "mean":
        _precompute_mean_inputs(dataset)
    dataloader = make_dataloader(dataset, args.batch_size, args.num_workers_test, args.seed)

    genes = [c[4:] for c in df.columns if c.startswith("rna_")]
    num_outputs = dataset.num_genes
    feature_dim = dataset.feature_dim

    pca_mu = None
    pca_U_k = None
    if args.head_type == "spex":
        pca_mu = np.zeros(num_outputs, dtype=np.float32)
        pca_U_k = np.zeros((num_outputs, int(args.spex_k)), dtype=np.float32)

    model = _build_model(
        args,
        model_type=args.model_type,
        num_outputs=num_outputs,
        feature_dim=feature_dim,
        device=device,
        pca_mu=pca_mu,
        pca_U_k=pca_U_k,
    )
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device)

    preds, labels, wsi_names, projects = evaluate(model, dataloader, run=None, suff="_full")

    if args.head_type == "spex":
        label_matrix = df[[c for c in df.columns if c.startswith("rna_")]].values.astype(np.float32)
        random_pca_mu, random_pca_u = make_random_spex_basis(
            num_outputs=num_outputs,
            spex_k=int(args.spex_k),
            seed=int(args.seed) + 1000,
            label_matrix=label_matrix,
        )
    else:
        random_pca_mu, random_pca_u = None, None

    random_model = _build_model(
        args,
        model_type=args.model_type,
        num_outputs=num_outputs,
        feature_dim=feature_dim,
        device=device,
        pca_mu=random_pca_mu,
        pca_U_k=random_pca_u,
    )
    random_model = random_model.to(device)
    random_preds, _, _, _ = evaluate(random_model, dataloader, run=None, suff="_full_rand")

    test_results = {
        "split_0": {
            "real": labels,
            "preds": preds,
            "random": random_preds,
            "wsi_file_name": wsi_names,
            "tcga_project": projects,
        },
        "genes": genes,
    }

    with open(os.path.join(save_dir, "test_results.pkl"), "wb") as f:
        pickle.dump(test_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    output_metadata = dict(source_metadata)
    output_metadata.update(
        {
            "experiment": args.exp_name,
            "cohort": args.cohort,
            "model_type": args.model_type,
            "head_type": args.head_type,
            "spex_k": int(args.spex_k),
            "num_folds": 1,
            "evaluation_mode": "full_dataset_test_only",
            "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
            "parameter_count": int(_backbone_param_count(model)),
        }
    )
    if args.metadata_json:
        output_metadata["source_metadata_json"] = str(Path(args.metadata_json).resolve())
    with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(output_metadata, f, indent=2, ensure_ascii=False)

    all_genes_df, sig_genes_df, summary_row = evaluate_predictions(
        preds=preds,
        labels=labels,
        random_preds=random_preds,
        genes=genes,
        experiment=args.exp_name,
        metadata=output_metadata,
    )
    all_genes_df.to_csv(os.path.join(save_dir, "all_genes.csv"))
    sig_genes_df.to_csv(os.path.join(save_dir, "sig_genes.csv"))
    pd.DataFrame([summary_row]).to_csv(os.path.join(save_dir, "summary_row.csv"), index=False)

    print(
        f"[{args.exp_name}] full-dataset evaluation finished: "
        f"all_mean_pcc={summary_row['all_mean_pcc']:.4f}, "
        f"top1000_mean_pcc={summary_row['top1000_mean_pcc']:.4f}, "
        f"sig_genes={summary_row['num_sig_genes']}"
    )
    print(f"Outputs saved to {save_dir}")


if __name__ == "__main__":
    main()
