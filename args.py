import argparse

import torch


def args_parser():
    parser = argparse.ArgumentParser(
        description="FedVHA federated learning experiments"
    )

    parser.add_argument(
        "--algorithm",
        choices=["fedavg", "fedvha", "fedprox", "fedawa"],
        default="fedvha",
    )
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "mnist", "svhn"],
        default="cifar10",
    )
    parser.add_argument("--E", type=int, default=5, help="local epochs")
    parser.add_argument("--r", type=int, default=300, help="communication rounds")
    parser.add_argument(
        "--K",
        type=int,
        default=100,
        help="number of clients",
    )
    parser.add_argument("--C", type=float, default=0.1, help="client sampling rate")
    parser.add_argument("--B", type=int, default=50, help="local batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="client learning rate")
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal strength")
    parser.add_argument("--fedawa_server_epochs", type=int, default=1)
    parser.add_argument("--fedawa_lr", type=float, default=1e-3)
    parser.add_argument("--fedawa_optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--fedawa_reg_distance", choices=["cos", "euc"], default="cos")
    parser.add_argument("--fedawa_gamma", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=10, help="local scheduler step size")
    parser.add_argument("--gamma", type=float, default=0.5, help="local scheduler decay")

    parser.add_argument(
        "--model",
        choices=["VGG16", "LeNet"],
        default="VGG16",
    )
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="PyTorch device, for example cpu or cuda:0",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")

    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--no_download", action="store_false", dest="download")
    parser.set_defaults(download=True)
    parser.add_argument("--beta", type=float, default=0.1, help="Dirichlet concentration")
    parser.add_argument("--min_client_samples", type=int, default=20)
    parser.add_argument("--server_val_per_class", type=int, default=200)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=1000)

    parser.add_argument("--hn_hidden_dim", type=int, default=64)
    parser.add_argument("--hn_lr", type=float, default=1e-3)
    parser.add_argument("--hn_temperature", type=float, default=1.5)
    parser.add_argument("--entropy_coef", type=float, default=1e-3)
    parser.add_argument("--kl_coef", type=float, default=1e-2)
    parser.add_argument("--hn_warmup_rounds", type=int, default=0)
    parser.add_argument("--hn_ramp_rounds", type=int, default=0)
    parser.add_argument("--hn_val_batches", type=int, default=1)
    parser.add_argument("--val_ema_momentum", type=float, default=0.9)
    parser.add_argument(
        "--hn_ablation",
        choices=[
            "full",
            "no_dynamic",
            "no_static",
            "no_regularization",
            "no_validation_feedback",
            "client_embedding",
            "client_scalar",
        ],
        default="full",
        help="FedVHA ablation variant",
    )

    parser.add_argument("--log_dir", default="./log_")
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument(
        "--test_interval",
        type=int,
        default=1,
        help="evaluate the untouched test set every N rounds",
    )

    args = parser.parse_args()
    if not 0 < args.C <= 1:
        parser.error("--C must be in (0, 1]")
    if args.beta <= 0:
        parser.error("--beta must be positive")
    if args.mu < 0:
        parser.error("--mu cannot be negative")
    if args.fedawa_server_epochs <= 0:
        parser.error("--fedawa_server_epochs must be positive")
    if args.fedawa_lr <= 0:
        parser.error("--fedawa_lr must be positive")
    if args.fedawa_gamma <= 0:
        parser.error("--fedawa_gamma must be positive")
    if args.E <= 0 or args.r <= 0 or args.B <= 0:
        parser.error("--E, --r, and --B must be positive")
    if args.hn_temperature <= 0:
        parser.error("--hn_temperature must be positive")
    if args.hn_val_batches <= 0:
        parser.error("--hn_val_batches must be positive")
    if not 0 <= args.val_ema_momentum < 1:
        parser.error("--val_ema_momentum must be in [0, 1)")
    if args.test_interval <= 0:
        parser.error("--test_interval must be positive")
    if args.checkpoint_interval < 0:
        parser.error("--checkpoint_interval cannot be negative")
    if args.dataset == "mnist" and args.model != "LeNet":
        parser.error("MNIST currently supports --model LeNet")
    if args.dataset == "svhn" and args.model != "VGG16":
        parser.error("SVHN currently supports VGG models")
    if args.dataset == "cifar10" and args.model != "VGG16":
        parser.error("CIFAR-10 currently supports VGG models")
    if args.algorithm == "fedvha" and int(args.C * args.K) < 2:
        parser.error("FedVHA needs at least two selected clients per round")
    if args.hn_ablation != "full" and args.algorithm != "fedvha":
        parser.error("--hn_ablation variants are only available with --algorithm fedvha")
    return args
