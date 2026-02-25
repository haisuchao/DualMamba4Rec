"""
DualMamba4Rec — Training Script

Sử dụng một file config duy nhất: configs/dualmamba4rec.yaml
Dataset-specific overrides được apply tự động trong script này.

Usage:
    python train.py --dataset diginetica
    python train.py --dataset tmall
    python train.py --dataset tafeng
    python train.py --dataset yoochoose1_64
    python train.py --dataset diginetica --lr 0.0005 --dropout 0.5
    python train.py --all
    python train.py --ablation no_gnn --dataset diginetica
"""

import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger

from model.dual_mamba4rec import DualMamba4Rec


# ── Unified config file ───────────────────────────────────────────────────────
CONFIG_FILE = 'configs/dualmamba4rec.yaml'

DATASETS = ['diginetica', 'tmall', 'tafeng', 'yoochoose1_64']

ABLATION_OVERRIDES = {
    'no_gnn':          {'gnn_layers': 0},
    'no_cross_attn':   {'n_heads': 0},
    'no_session_attn': {'use_session_attention': False},
    'gnn_only':        {'n_layers': 0},
}


# ── Core run function ─────────────────────────────────────────────────────────

def run(dataset: str, extra_params: dict = None):
    """
    Train và evaluate DualMamba4Rec trên một dataset.

    Args:
        dataset     : tên dataset
        extra_params: dict params để override config (CLI args hoặc ablation)
    Returns:
        dict với best_valid và test results
    """
    print(f"\n{'═'*60}")
    print(f"  DualMamba4Rec — {dataset.upper()}")
    print(f"{'═'*60}\n")

    # Gộp: default config ← extra_params (CLI / ablation)
    params = {'dataset': dataset}
    if extra_params:
        params.update(extra_params)

    config = Config(
        model=DualMamba4Rec,
        dataset=dataset,
        config_file_list=[CONFIG_FILE],
        config_dict=params,
    )

    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = logging.getLogger()
    logger.info(f"Config:\n{config}")

    # Dataset & DataLoader
    print("[1/4] Loading dataset...")
    dataset_obj = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset_obj)

    # Model
    print("[2/4] Building model...")
    model = DualMamba4Rec(config, dataset_obj).to(config['device'])
    logger.info(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Trainable parameters: {n_params:,}")

    # Build global graph
    print("[3/4] Building item transition graph...")
    model.pre_build_graph(dataset_obj)

    # Train
    print("[4/4] Training...")
    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data,
        saved=True, show_progress=config['show_progress'],
    )

    # Test
    test_result = trainer.evaluate(
        test_data, load_best_model=True,
        show_progress=config['show_progress'],
    )

    _print_results(dataset, test_result)
    logger.info(f"Best valid: {best_valid_result}")
    logger.info(f"Test:       {test_result}")

    return {'dataset': dataset, 'best_valid': best_valid_result, 'test': test_result}


def _print_results(dataset: str, result: dict):
    print(f"\n{'─'*50}")
    print(f"  {dataset.upper()} — Test Results")
    print(f"{'─'*50}")
    for k, v in sorted(result.items()):
        print(f"  {k:25s}: {v:.4f}")
    print(f"{'─'*50}\n")


def _summary_table(all_results: dict):
    print(f"\n{'═'*65}")
    print(f"  SUMMARY")
    print(f"{'═'*65}")
    print(f"  {'Dataset':15s} {'R@20':>8} {'M@20':>8} {'R@15':>8} {'M@15':>8}")
    print(f"  {'─'*55}")
    for dataset, res in all_results.items():
        if res is None:
            continue
        tr = res.get('test', {})
        def g(k): return tr.get(k, tr.get(k.title(), 0.0))
        print(f"  {dataset:15s} {g('recall@20'):>8.4f} {g('mrr@20'):>8.4f}"
              f" {g('recall@15'):>8.4f} {g('mrr@15'):>8.4f}")
    print(f"{'═'*65}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='DualMamba4Rec Training',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--dataset', type=str, default='diginetica',
                        choices=DATASETS, help='Dataset to train on')
    parser.add_argument('--all', action='store_true',
                        help='Train on all datasets sequentially')
    parser.add_argument('--ablation', type=str, default=None,
                        choices=list(ABLATION_OVERRIDES.keys()) + ['all'],
                        help='Run ablation variant')

    # Hyperparameter overrides
    parser.add_argument('--lr',         type=float, help='Learning rate')
    parser.add_argument('--dropout',    type=float, help='Dropout rate')
    parser.add_argument('--d_model',    type=int,   help='Embedding dimension')
    parser.add_argument('--n_layers',   type=int,   help='C-Mamba layers')
    parser.add_argument('--gnn_layers', type=int,   help='GNN LightGCN hops')
    parser.add_argument('--n_heads',    type=int,   help='Cross-attention heads')
    parser.add_argument('--batch_size', type=int,   help='Training batch size')

    args = parser.parse_args()

    # Collect CLI overrides (chỉ những gì user thực sự chỉ định)
    cli_params = {}
    if args.lr         is not None: cli_params['learning_rate']    = args.lr
    if args.dropout    is not None: cli_params['dropout_prob']      = args.dropout
    if args.d_model    is not None: cli_params['embedding_size']    = args.d_model
    if args.n_layers   is not None: cli_params['n_layers']          = args.n_layers
    if args.gnn_layers is not None: cli_params['gnn_layers']        = args.gnn_layers
    if args.n_heads    is not None: cli_params['n_heads']           = args.n_heads
    if args.batch_size is not None: cli_params['train_batch_size']  = args.batch_size

    # ── Run ───────────────────────────────────────────────────────────────────
    if args.all:
        all_results = {}
        for ds in DATASETS:
            try:
                all_results[ds] = run(ds, cli_params or None)
            except Exception as e:
                print(f"[ERROR] {ds} failed: {e}")
                all_results[ds] = None
        _summary_table(all_results)

    elif args.ablation == 'all':
        all_results = {}
        # Full model trước
        all_results['full'] = run(args.dataset, cli_params or None)
        # Từng ablation
        for name, overrides in ABLATION_OVERRIDES.items():
            try:
                params = {**cli_params, **overrides}
                all_results[name] = run(args.dataset, params)
            except Exception as e:
                print(f"[ERROR] Ablation {name} failed: {e}")
                all_results[name] = None
        _summary_table(all_results)

    elif args.ablation is not None:
        params = {**cli_params, **ABLATION_OVERRIDES[args.ablation]}
        run(args.dataset, params)

    else:
        run(args.dataset, cli_params or None)


if __name__ == '__main__':
    main()
