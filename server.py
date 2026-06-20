import copy
import logging
import math
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

from client import evaluate_batch, test, train
from get_data import (
    build_federated_cifar10,
    build_federated_mnist,
    build_federated_svhn,
)
from hypernetwork import AggregationHyperNetwork
from model import build_model


DYNAMIC_FEATURE_NAMES = (
    "final_loss",
    "loss_drop",
    "delta_norm",
    "delta_norm_ratio",
    "client_val_loss",
    "client_val_acc",
    "val_loss_gain",
    "val_acc_gain",
    "ema_val_loss_gain",
    "ema_val_acc_gain",
)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FedAvg:
    """Runs either sample-size FedAvg or validation-trained HyperFedAvg."""

    def __init__(self, args):
        self.args = args
        self.args.device = torch.device(args.device)
        if self.args.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {self.args.device}")

        set_random_seed(args.seed)
        if args.dataset == "mnist":
            self.data = build_federated_mnist(args)
        elif args.dataset == "svhn":
            self.data = build_federated_svhn(args)
        else:
            self.data = build_federated_cifar10(args)
        if len(self.data.train_loaders) != args.K:
            raise RuntimeError(
                f"Expected {args.K} client loaders, got {len(self.data.train_loaders)}"
            )

        self.model = build_model(args.model, args.dataset).to(args.device)
        # One reusable local model avoids allocating 100 copies of VGG.
        self.client_model = copy.deepcopy(self.model).to(args.device)
        self.param_names = tuple(name for name, _ in self.model.named_parameters())
        self.param_name_set = set(self.param_names)
        sample_fractions = torch.tensor(self.data.client_sizes, dtype=torch.float32)
        sample_fractions = sample_fractions / sample_fractions.sum()
        self.fedawa_logits = torch.log(sample_fractions + 1e-12)

        self.dynamic_feature_dim = len(DYNAMIC_FEATURE_NAMES)
        self.static_feature_dim = self.data.client_features.shape[1]
        self.hypernetwork = None
        self.client_embedding = None
        self.client_embedding_head = None
        self.client_scalar_logits = None
        self.hn_optimizer = None
        if (
            args.algorithm == "hyperfedavg"
            and args.hn_ablation != "no_validation_feedback"
        ):
            if args.hn_ablation == "client_scalar":
                self.client_scalar_logits = torch.nn.Parameter(
                    torch.log(sample_fractions + 1e-12).to(args.device)
                )
                self.hn_optimizer = torch.optim.Adam(
                    [self.client_scalar_logits],
                    lr=args.hn_lr,
                )
            elif args.hn_ablation == "client_embedding":
                self.client_embedding = torch.nn.Embedding(
                    args.K,
                    args.hn_hidden_dim,
                ).to(args.device)
                self.client_embedding_head = torch.nn.Linear(
                    args.hn_hidden_dim,
                    1,
                ).to(args.device)
                self.hn_optimizer = torch.optim.Adam(
                    list(self.client_embedding.parameters())
                    + list(self.client_embedding_head.parameters()),
                    lr=args.hn_lr,
                )
            else:
                feature_dim = self._hypernetwork_feature_dim()
                self.hypernetwork = AggregationHyperNetwork(
                    feature_dim=feature_dim,
                    hidden_dim=args.hn_hidden_dim,
                ).to(args.device)
                self.hn_optimizer = torch.optim.Adam(
                    self.hypernetwork.parameters(),
                    lr=args.hn_lr,
                )

        self.client_val_ema = {}
        self.val_iter = iter(self.data.server_val_loader)
        self.logger = None
        self.run_dir = None

    def server(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = self._create_logger(timestamp)
        clients_per_round = max(int(self.args.C * self.args.K), 1)
        if self.args.algorithm == "hyperfedavg" and clients_per_round < 2:
            raise ValueError("HyperFedAvg requires at least two clients per round")

        self.logger.info("=== Federated training configuration ===")
        self.logger.info(
            "algorithm=%s dataset=%s model=%s clients=%d clients_per_round=%d",
            self.args.algorithm,
            self.args.dataset,
            self.args.model,
            self.args.K,
            clients_per_round,
        )
        self.logger.info(
            "rounds=%d local_epochs=%d batch_size=%d client_lr=%g beta=%g seed=%d",
            self.args.r,
            self.args.E,
            self.args.B,
            self.args.lr,
            self.args.beta,
            self.args.seed,
        )
        if self.args.algorithm == "fedprox":
            self.logger.info("fedprox_mu=%g", self.args.mu)
        if self.args.algorithm == "fedlc":
            self.logger.info("fedlc_tau=%g", self.args.fedlc_tau)
        if self.args.algorithm == "fedawa":
            self.logger.info(
                "fedawa_server_epochs=%d fedawa_lr=%g fedawa_optimizer=%s "
                "fedawa_reg_distance=%s fedawa_gamma=%g",
                self.args.fedawa_server_epochs,
                self.args.fedawa_lr,
                self.args.fedawa_optimizer,
                self.args.fedawa_reg_distance,
                self.args.fedawa_gamma,
            )
        self.logger.info(
            "client_train_samples=%d server_validation_samples=%d test_samples=%d",
            sum(self.data.client_sizes),
            len(self.data.server_val_indices),
            len(self.data.test_loader.dataset),
        )
        if (
            self.hypernetwork is not None
            or self.client_embedding is not None
            or self.client_scalar_logits is not None
        ):
            self.logger.info(
                "hn_lr=%g temperature=%g warmup=%d ramp=%d ablation=%s "
                "input_dim=%d static_features=%s dynamic_features=%s",
                self.args.hn_lr,
                self.args.hn_temperature,
                self.args.hn_warmup_rounds,
                self.args.hn_ramp_rounds,
                self.args.hn_ablation,
                self._hypernetwork_feature_dim(),
                self._uses_static_features(),
                self._uses_dynamic_features(),
                ",".join(DYNAMIC_FEATURE_NAMES),
            )
        elif self.args.algorithm == "hyperfedavg":
            self.logger.info(
                "hn_ablation=%s uses fixed FedAvg-style sample aggregation",
                self.args.hn_ablation,
            )

        best_val_acc = -1.0
        best_checkpoint = None
        latest_test_loss = float("nan")
        latest_test_acc = float("nan")

        for round_num in tqdm(range(1, self.args.r + 1), desc="Communication rounds"):
            selected_clients = random.sample(
                range(self.args.K),
                clients_per_round,
            )
            client_results = self.client_update(selected_clients, round_num)
            aggregate_info = self.server_aggregate(
                selected_clients,
                client_results,
                round_num,
            )

            val_loss, val_acc = test(
                self.args,
                self.model,
                self.data.server_val_eval_loader,
            )
            if round_num % self.args.test_interval == 0 or round_num == self.args.r:
                latest_test_loss, latest_test_acc = test(
                    self.args,
                    self.model,
                    self.data.test_loader,
                )

            self.logger.info(
                "round=%d selected=%s val_loss=%.6f val_acc=%.6f "
                "test_loss=%.6f test_acc=%.6f hn_loss=%.6f kl_loss=%.6f "
                "fedawa_loss=%.6f "
                "hn_alpha=%.4f delta_norm=%.6f weight_std=%.6f "
                "raw_weights=%s weights=%s",
                round_num,
                selected_clients,
                val_loss,
                val_acc,
                latest_test_loss,
                latest_test_acc,
                aggregate_info["hn_loss"],
                aggregate_info["kl_loss"],
                aggregate_info["fedawa_loss"],
                aggregate_info["hn_alpha"],
                aggregate_info["delta_g_norm"],
                aggregate_info["weight_std"],
                [round(value, 6) for value in aggregate_info["raw_weights"]],
                [round(value, 6) for value in aggregate_info["weights"]],
            )

            # The test set is never used for model selection.
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_checkpoint = self.save_checkpoint(
                    f"best_round_{round_num}",
                    round_num,
                    {
                        "validation_loss": val_loss,
                        "validation_accuracy": val_acc,
                        "test_loss_at_save": latest_test_loss,
                        "test_accuracy_at_save": latest_test_acc,
                    },
                )

            if (
                self.args.checkpoint_interval > 0
                and round_num % self.args.checkpoint_interval == 0
            ):
                self.save_checkpoint(
                    f"checkpoint_round_{round_num}",
                    round_num,
                    {
                        "validation_loss": val_loss,
                        "validation_accuracy": val_acc,
                    },
                )

        self.save_checkpoint(
            "final_model",
            self.args.r,
            {
                "best_validation_accuracy": best_val_acc,
                "best_checkpoint": best_checkpoint,
                "final_test_loss": latest_test_loss,
                "final_test_accuracy": latest_test_acc,
            },
        )
        self.logger.info(
            "training_complete best_val_acc=%.6f best_checkpoint=%s",
            best_val_acc,
            best_checkpoint,
        )
        return self.model

    def client_update(self, selected_clients, round_num):
        feature_data, feature_target = self._next_validation_batch()
        global_val_loss, global_val_acc = evaluate_batch(
            self.model,
            feature_data,
            feature_target,
        )
        global_state_cpu = self._snapshot_state(self.model)
        global_state_device = self.model.state_dict()
        results = []

        for client_id in selected_clients:
            client_seed = (
                self.args.seed
                + round_num * self.args.K
                + client_id
            )
            torch.manual_seed(client_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(client_seed)
                torch.cuda.manual_seed_all(client_seed)
            self.client_model.load_state_dict(global_state_device, strict=True)
            train_kwargs = {"client_id": client_id}
            if self.args.algorithm == "fedprox":
                train_kwargs["global_state"] = global_state_cpu
            if self.args.algorithm == "fedlc":
                train_kwargs["class_counts"] = self.data.client_class_counts[
                    client_id
                ].to(self.args.device)
            train_stats = train(
                self.args,
                self.client_model,
                self.data.train_loaders[client_id],
                **train_kwargs,
            )
            client_val_loss, client_val_acc = evaluate_batch(
                self.client_model,
                feature_data,
                feature_target,
            )
            local_state_cpu = self._snapshot_state(self.client_model)
            delta_norm = self._parameter_delta_norm(
                global_state_cpu,
                local_state_cpu,
            )

            val_loss_gain = global_val_loss - client_val_loss
            val_acc_gain = client_val_acc - global_val_acc
            previous = self.client_val_ema.get(client_id)
            if previous is None:
                ema_loss_gain = val_loss_gain
                ema_acc_gain = val_acc_gain
            else:
                momentum = self.args.val_ema_momentum
                ema_loss_gain = (
                    momentum * previous["ema_val_loss_gain"]
                    + (1.0 - momentum) * val_loss_gain
                )
                ema_acc_gain = (
                    momentum * previous["ema_val_acc_gain"]
                    + (1.0 - momentum) * val_acc_gain
                )
            self.client_val_ema[client_id] = {
                "ema_val_loss_gain": ema_loss_gain,
                "ema_val_acc_gain": ema_acc_gain,
            }

            results.append(
                {
                    "client_id": client_id,
                    "state": local_state_cpu,
                    "delta_norm": delta_norm,
                    **train_stats,
                    "client_val_loss": client_val_loss,
                    "client_val_acc": client_val_acc,
                    "val_loss_gain": val_loss_gain,
                    "val_acc_gain": val_acc_gain,
                    "ema_val_loss_gain": ema_loss_gain,
                    "ema_val_acc_gain": ema_acc_gain,
                }
            )

        return {
            "global_state": global_state_cpu,
            "clients": results,
            "feature_global_val_loss": global_val_loss,
            "feature_global_val_acc": global_val_acc,
        }

    def server_aggregate(self, selected_clients, client_results, round_num):
        sample_counts = torch.tensor(
            [self.data.client_sizes[client_id] for client_id in selected_clients],
            dtype=torch.float32,
            device=self.args.device,
        )
        fedavg_weights = sample_counts / sample_counts.sum()

        if self.args.algorithm == "fedawa":
            raw_weights, fedawa_loss_value = self._fedawa_weights(
                selected_clients,
                client_results,
            )
            aggregation_weights = raw_weights
            hn_alpha = 0.0
            with torch.no_grad():
                new_state, delta_g_norm = self._weighted_state(
                    client_results["global_state"],
                    client_results["clients"],
                    aggregation_weights,
                    update_scale=self.args.fedawa_gamma,
                )
            hn_loss_value = 0.0
            kl_loss_value = 0.0
        elif (
            self.hypernetwork is None
            and self.client_embedding is None
            and self.client_scalar_logits is None
        ):
            aggregation_weights = fedavg_weights
            hn_alpha = 0.0
            with torch.no_grad():
                new_state, delta_g_norm = self._weighted_state(
                    client_results["global_state"],
                    client_results["clients"],
                    aggregation_weights,
                )
            hn_loss_value = 0.0
            kl_loss_value = 0.0
            fedawa_loss_value = 0.0
            raw_weights = fedavg_weights
        else:
            logits = self._aggregation_logits(
                selected_clients,
                client_results["clients"],
            )
            raw_weights = F.softmax(
                logits / self.args.hn_temperature,
                dim=0,
            )

            if round_num <= self.args.hn_warmup_rounds:
                hn_alpha = 0.0
            else:
                hn_alpha = min(
                    1.0,
                    (round_num - self.args.hn_warmup_rounds)
                    / max(self.args.hn_ramp_rounds, 1),
                )
            aggregation_weights = (
                (1.0 - hn_alpha) * fedavg_weights
                + hn_alpha * raw_weights
            )

            new_state, delta_g_norm = self._weighted_state(
                client_results["global_state"],
                client_results["clients"],
                aggregation_weights,
            )
            validation_loss = self._functional_validation_loss(new_state)
            entropy = -(
                raw_weights * torch.log(raw_weights + 1e-8)
            ).sum()
            kl_loss = (
                raw_weights
                * (
                    torch.log(raw_weights + 1e-8)
                    - torch.log(fedavg_weights + 1e-8)
                )
            ).sum()
            entropy_coef = self.args.entropy_coef
            kl_coef = self.args.kl_coef
            if self.args.hn_ablation == "no_regularization":
                entropy_coef = 0.0
                kl_coef = 0.0
            hn_loss = validation_loss - entropy_coef * entropy + kl_coef * kl_loss

            self.hn_optimizer.zero_grad(set_to_none=True)
            hn_loss.backward()
            self.hn_optimizer.step()
            hn_loss_value = hn_loss.item()
            kl_loss_value = kl_loss.item()
            fedawa_loss_value = 0.0

        current_state = self.model.state_dict()
        detached_state = {
            name: value.detach().to(
                device=current_state[name].device,
                dtype=current_state[name].dtype,
            )
            for name, value in new_state.items()
        }
        self.model.load_state_dict(detached_state, strict=True)

        return {
            "hn_loss": hn_loss_value,
            "kl_loss": kl_loss_value,
            "fedawa_loss": fedawa_loss_value,
            "hn_alpha": hn_alpha,
            "delta_g_norm": delta_g_norm,
            "raw_weights": raw_weights.detach().cpu().tolist(),
            "weights": aggregation_weights.detach().cpu().tolist(),
            "weight_std": aggregation_weights.detach().std(unbiased=False).item(),
        }

    def _fedawa_weights(self, selected_clients, client_results):
        selected_logits = (
            self.fedawa_logits[selected_clients]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        if self.args.fedawa_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                [selected_logits],
                lr=self.args.fedawa_lr,
                momentum=0.9,
                weight_decay=5e-4,
            )
        else:
            optimizer = torch.optim.Adam(
                [selected_logits],
                lr=self.args.fedawa_lr,
                betas=(0.5, 0.999),
            )

        global_vector = self._flatten_params(client_results["global_state"])
        local_vectors = torch.stack(
            [
                self._flatten_params(result["state"])
                for result in client_results["clients"]
            ],
            dim=0,
        )
        client_updates = local_vectors - global_vector.unsqueeze(0)

        fedawa_loss = torch.zeros((), dtype=torch.float32)
        for _ in range(self.args.fedawa_server_epochs):
            probabilities = F.softmax(selected_logits, dim=0)

            if self.args.fedawa_reg_distance == "cos":
                distance_to_global = 1.0 - F.cosine_similarity(
                    local_vectors,
                    global_vector.unsqueeze(0),
                    dim=1,
                    eps=1e-8,
                )
            else:
                distance_to_global = (
                    local_vectors - global_vector.unsqueeze(0)
                ).abs().square().mean(dim=1)
            reg_loss = (probabilities * distance_to_global.detach()).sum()

            weighted_update = torch.matmul(
                probabilities.unsqueeze(0),
                client_updates.detach(),
            ).squeeze(0)
            update_distance = torch.norm(
                client_updates.detach() - weighted_update.unsqueeze(0),
                p=2,
                dim=1,
            )
            sim_loss = (probabilities * update_distance).sum()
            fedawa_loss = reg_loss + sim_loss

            optimizer.zero_grad(set_to_none=True)
            fedawa_loss.backward()
            optimizer.step()

        with torch.no_grad():
            self.fedawa_logits[selected_clients] = selected_logits.detach()
            weights = F.softmax(selected_logits.detach(), dim=0).to(self.args.device)
        return weights, fedawa_loss.detach().item()

    def _flatten_params(self, state):
        return torch.cat(
            [
                state[name].detach().float().reshape(-1).cpu()
                for name in self.param_names
            ],
            dim=0,
        )

    def _uses_static_features(self):
        return self.args.hn_ablation not in {
            "no_static",
            "client_embedding",
            "client_scalar",
        }

    def _uses_dynamic_features(self):
        return self.args.hn_ablation not in {
            "no_dynamic",
            "client_embedding",
            "client_scalar",
        }

    def _hypernetwork_feature_dim(self):
        if self.args.hn_ablation == "client_embedding":
            return self.args.hn_hidden_dim
        if self.args.hn_ablation == "client_scalar":
            return 1
        feature_dim = 0
        if self._uses_static_features():
            feature_dim += self.static_feature_dim
        if self._uses_dynamic_features():
            feature_dim += self.dynamic_feature_dim
        return feature_dim

    def _build_hypernetwork_features(self, selected_clients, client_results):
        features = []
        if self._uses_static_features():
            features.append(self.data.client_features[selected_clients].to(self.args.device))
        if self._uses_dynamic_features():
            features.append(self._build_dynamic_features(client_results))
        if not features:
            raise RuntimeError("HyperFedAvg needs at least one feature group")
        return torch.cat(features, dim=1)

    def _aggregation_logits(self, selected_clients, client_results):
        if self.args.hn_ablation == "client_scalar":
            client_ids = torch.tensor(
                selected_clients,
                dtype=torch.long,
                device=self.args.device,
            )
            return self.client_scalar_logits[client_ids]

        if self.args.hn_ablation == "client_embedding":
            client_ids = torch.tensor(
                selected_clients,
                dtype=torch.long,
                device=self.args.device,
            )
            embeddings = self.client_embedding(client_ids)
            return self.client_embedding_head(embeddings).squeeze(-1)

        feature_matrix = self._build_hypernetwork_features(
            selected_clients,
            client_results,
        )
        return self.hypernetwork(feature_matrix)

    def _weighted_state(self, global_state, client_results, weights, update_scale=1.0):
        new_state = {}
        delta_square_sum = 0.0
        # Hypernetwork gradients must flow through trainable parameters only.
        # BatchNorm running statistics are buffers, and cuDNN does not support
        # differentiating batch_norm with respect to running_mean/running_var.
        buffer_weights = weights.detach()

        for name, global_value_cpu in global_state.items():
            global_value = global_value_cpu.to(self.args.device)
            local_values = [
                result["state"][name].to(self.args.device)
                for result in client_results
            ]

            if name in self.param_name_set:
                aggregate_delta = torch.zeros_like(global_value)
                for position, local_value in enumerate(local_values):
                    aggregate_delta = aggregate_delta + weights[position] * (
                        local_value.to(global_value.dtype) - global_value
                    )
                aggregate_delta = update_scale * aggregate_delta
                new_state[name] = global_value + aggregate_delta
                delta_square_sum += aggregate_delta.detach().float().square().sum().item()
            elif torch.is_floating_point(global_value):
                aggregated_buffer = torch.zeros_like(global_value)
                for position, local_value in enumerate(local_values):
                    aggregated_buffer = (
                        aggregated_buffer
                        + buffer_weights[position]
                        * local_value.to(global_value.dtype)
                    )
                new_state[name] = aggregated_buffer.detach()
            else:
                # BatchNorm num_batches_tracked is an integer buffer.
                new_state[name] = (
                    torch.stack(local_values, dim=0)
                    .max(dim=0)
                    .values.detach()
                )

        return new_state, math.sqrt(delta_square_sum)

    def _functional_validation_loss(self, state):
        differentiable_buffers = [
            name
            for name, value in state.items()
            if name not in self.param_name_set and value.requires_grad
        ]
        if differentiable_buffers:
            raise RuntimeError(
                "Model buffers unexpectedly require gradients: "
                + ", ".join(differentiable_buffers[:5])
            )

        was_training = self.model.training
        self.model.eval()
        losses = []
        for _ in range(self.args.hn_val_batches):
            data, target = self._next_validation_batch()
            logits = functional_call(self.model, state, (data,))
            losses.append(F.cross_entropy(logits, target.long()))
        if was_training:
            self.model.train()
        return torch.stack(losses).mean()

    def _build_dynamic_features(self, client_results):
        mean_delta_norm = np.mean(
            [result["delta_norm"] for result in client_results]
        ) + 1e-8
        rows = []
        for result in client_results:
            rows.append(
                [
                    result["final_loss"],
                    result["loss_drop"],
                    result["delta_norm"],
                    result["delta_norm"] / mean_delta_norm,
                    result["client_val_loss"],
                    result["client_val_acc"],
                    result["val_loss_gain"],
                    result["val_acc_gain"],
                    result["ema_val_loss_gain"],
                    result["ema_val_acc_gain"],
                ]
            )
        features = torch.tensor(
            rows,
            dtype=torch.float32,
            device=self.args.device,
        )
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True, unbiased=False)
        return (features - mean) / (std + 1e-8)

    def _next_validation_batch(self):
        try:
            data, target = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.data.server_val_loader)
            data, target = next(self.val_iter)
        return (
            data.to(self.args.device, non_blocking=self.args.pin_memory),
            target.to(self.args.device, non_blocking=self.args.pin_memory),
        )

    def _parameter_delta_norm(self, global_state, local_state):
        square_sum = 0.0
        for name in self.param_names:
            delta = local_state[name].float() - global_state[name].float()
            square_sum += delta.square().sum().item()
        return math.sqrt(square_sum)

    @staticmethod
    def _snapshot_state(model):
        return {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }

    def save_checkpoint(self, prefix, round_num, metadata=None):
        checkpoint_dir = self.run_dir or self.args.log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"{prefix}.pth")
        checkpoint = {
            "round": round_num,
            "algorithm": self.args.algorithm,
            "args": vars(self.args),
            "global_model": self.model.state_dict(),
            "metadata": metadata or {},
        }
        if self.hypernetwork is not None:
            checkpoint["hypernetwork"] = self.hypernetwork.state_dict()
            checkpoint["hn_optimizer"] = self.hn_optimizer.state_dict()
        if self.client_embedding is not None:
            checkpoint["client_embedding"] = self.client_embedding.state_dict()
            checkpoint["client_embedding_head"] = self.client_embedding_head.state_dict()
            checkpoint["hn_optimizer"] = self.hn_optimizer.state_dict()
        if self.client_scalar_logits is not None:
            checkpoint["client_scalar_logits"] = self.client_scalar_logits.detach().cpu()
            checkpoint["hn_optimizer"] = self.hn_optimizer.state_dict()
        if self.args.algorithm == "fedawa":
            checkpoint["fedawa_logits"] = self.fedawa_logits.detach().cpu()
        torch.save(checkpoint, path)
        return path

    def _create_logger(self, timestamp):
        self.run_dir = os.path.join(
            self.args.log_dir,
            f"{self.args.algorithm}_{timestamp}",
        )
        os.makedirs(self.run_dir, exist_ok=True)
        log_path = os.path.join(self.run_dir, "training.log")
        logger = logging.getLogger(f"federated.{timestamp}")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger
