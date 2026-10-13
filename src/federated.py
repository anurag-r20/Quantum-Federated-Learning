"""
Flower simulation of federated learning on the brain-tumour MRI dataset, in plaintext or
with every model update CKKS-encrypted (TenSEAL).

Threat model: an honest-but-curious server. In `capture_round` the server asks clients for a
single SGD step on a small batch (FedSGD, the setting studied by DLG / iDLG / Geiping et al.)
and records exactly what it receives. The attacks in gia_attack.py run on that recording.
With FHE on, the server only holds the public context: it aggregates ciphertexts but can
never decrypt an individual update (or the aggregate).
"""
import functools
import json
import logging
import os
import pickle
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import tenseal as ts
import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, GetParametersIns, GetParametersRes, Parameters
from flwr.common.logger import logger as flwr_logger
from flwr.common.typing import Code, Status
from torch.utils.data import DataLoader, TensorDataset

from models import MRINet
from utils import data_setup, engine, security
from utils.common import get_parameters2, set_parameters

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OK = Status(code=Code.OK, message="")


@dataclass
class FLConfig:
    he: bool = False                  # encrypt every client update with CKKS
    data_path: str = "data/MRI"
    img_size: int = 64
    num_clients: int = 3
    partition: str = "iid"            # 'iid' or 'non_iid'
    rounds: int = 10
    local_epochs: int = 2
    batch_size: int = 32
    lr: float = 1e-3                  # Adam, normal FedAvg rounds
    capture_round: int = 7            # round in which clients send a single-batch gradient step
    capture_batch_size: int = 1
    capture_lr: float = 0.01          # SGD step size in the capture round (known to the server)
    val_fraction: float = 0.1
    seed: int = 0
    out_dir: str = "results/plaintext"
    keys_dir: str = "keys"
    cpus_per_client: int = 4


# ------------------------------------------------------------------ keys

def ensure_keys(keys_dir="keys"):
    """Create (once) the shared CKKS key pair: secret context for clients, public context for the server."""
    os.makedirs(keys_dir, exist_ok=True)
    secret_path = os.path.join(keys_dir, "client_secret_context.pkl")
    public_path = os.path.join(keys_dir, "server_public_context.pkl")
    if not (os.path.exists(secret_path) and os.path.exists(public_path)):
        ctx = security.context()
        security.write_query(secret_path, {"contexte": ctx.serialize(save_secret_key=True)})
        security.write_query(public_path, {"contexte": ctx.serialize(save_secret_key=False)})
    return secret_path, public_path


@functools.lru_cache(maxsize=4)
def load_context(path):
    _, serialized = security.read_query(path)
    return ts.context_from(serialized)


# ------------------------------------------------------------------ data

@functools.lru_cache(maxsize=2)
def client_splits(data_path, img_size, num_clients, partition, seed, val_fraction):
    """Per-client (train_indices, val_indices) into the MRI training set."""
    arrays = data_setup.load_mri_arrays(data_path, img_size)
    class_to_idx = {c: i for i, c in enumerate(arrays["classes"])}
    parts = data_setup.partition_indices(arrays["y_train"], num_clients, partition, seed, class_to_idx)
    splits = []
    for cid, idx in enumerate(parts):
        idx = np.random.default_rng(seed + cid).permutation(idx)
        n_val = int(len(idx) * val_fraction)
        splits.append((idx[n_val:], idx[:n_val]))
    return arrays, splits


def client_dataset(cfg: FLConfig, cid: int, split: str = "train") -> TensorDataset:
    arrays, splits = client_splits(cfg.data_path, cfg.img_size, cfg.num_clients, cfg.partition, cfg.seed,
                                   cfg.val_fraction)
    idx = splits[cid][0 if split == "train" else 1]
    return TensorDataset(data_setup.to_model_input(arrays["x_train"][idx]), torch.as_tensor(arrays["y_train"][idx]))


# ------------------------------------------------------------------ client

class MRIClient(fl.client.Client):
    """A hospital: trains locally, sends plaintext or CKKS-encrypted weights."""

    def __init__(self, cid: str, cfg: FLConfig):
        torch.set_num_threads(cfg.cpus_per_client)
        self.cid, self.cfg = int(cid), cfg
        self.train_ds = client_dataset(cfg, self.cid, "train")
        self.val_ds = client_dataset(cfg, self.cid, "val")
        self.model = MRINet(num_classes=4, img_size=cfg.img_size)
        self.ctx = load_context(ensure_keys(cfg.keys_dir)[0]) if cfg.he else None

    def _load_global(self, parameters: Parameters):
        weights = security.parameters_to_weights(parameters, self.ctx)
        encrypted = parameters.tensor_type == security.TENSOR_TYPE_CKKS
        set_parameters(self.model, weights, self.ctx if encrypted else None, save_path=self.cfg.out_dir)

    def _export(self) -> Parameters:
        return security.weights_to_parameters(get_parameters2(self.model, self.ctx, save_path=self.cfg.out_dir))

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return GetParametersRes(status=OK, parameters=self._export())

    def fit(self, ins: FitIns) -> FitRes:
        self._load_global(ins.parameters)
        config = ins.config
        server_round = int(config["server_round"])
        torch.manual_seed(self.cfg.seed + 1000 * server_round + self.cid)
        loss_fn = torch.nn.CrossEntropyLoss()

        if config["fedsgd"]:
            # One SGD step on a small batch: the update is exactly -lr * gradient.
            batch = torch.randperm(len(self.train_ds))[:int(config["batch_size"])]
            images, labels = self.train_ds.tensors[0][batch], self.train_ds.tensors[1][batch]
            optimizer = torch.optim.SGD(self.model.parameters(), lr=float(config["lr"]))
            optimizer.zero_grad()
            loss = loss_fn(self.model(images), labels)
            loss.backward()
            optimizer.step()
            # Saved for *evaluating* the attack only; the attacker never reads this file.
            gt_dir = os.path.join(self.cfg.out_dir, "gia_ground_truth")
            os.makedirs(gt_dir, exist_ok=True)
            torch.save({"images": images, "labels": labels, "round": server_round},
                       os.path.join(gt_dir, f"client{self.cid}.pt"))
            num_examples, metrics = len(batch), {"train_loss": float(loss)}
        else:
            loader = DataLoader(self.train_ds, batch_size=int(config["batch_size"]), shuffle=True)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=float(config["lr"]))
            for _ in range(int(config["local_epochs"])):
                train_loss, train_acc = engine.train_step(self.model, loader, loss_fn, optimizer, "cpu")
            num_examples, metrics = len(self.train_ds), {"train_loss": train_loss, "train_acc": train_acc}

        return FitRes(status=OK, parameters=self._export(), num_examples=num_examples, metrics=metrics)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        self._load_global(ins.parameters)
        loss, accuracy, *_ = engine.test(self.model, DataLoader(self.val_ds, batch_size=256),
                                         torch.nn.CrossEntropyLoss(), "cpu")
        return EvaluateRes(status=OK, loss=float(loss), num_examples=len(self.val_ds),
                           metrics={"accuracy": float(accuracy)})


def _client_fn(cid: str, cfg: FLConfig) -> MRIClient:
    return MRIClient(cid, cfg)


# ------------------------------------------------------------------ server

class CapturingFedAvg(fl.server.strategy.Strategy):
    """FedAvg over plaintext or encrypted weights; records what the server sees in the capture round."""

    def __init__(self, cfg: FLConfig, public_ctx=None):
        super().__init__()
        self.cfg, self.public_ctx = cfg, public_ctx
        self.history = []
        self._round_start = None
        self._sent_in_capture_round = None

    def __repr__(self):
        return f"CapturingFedAvg(he={self.cfg.he})"

    def initialize_parameters(self, client_manager):
        torch.manual_seed(self.cfg.seed)
        arrays = [v.numpy() for v in MRINet(num_classes=4, img_size=self.cfg.img_size).state_dict().values()]
        if self.cfg.he:  # the server encrypts the initial model with the public key
            return security.weights_to_parameters([security.encrypt_array(self.public_ctx, a) for a in arrays])
        return security.weights_to_parameters(arrays)

    def _fit_config(self, server_round):
        if server_round == self.cfg.capture_round:
            return {"server_round": server_round, "fedsgd": True,
                    "batch_size": self.cfg.capture_batch_size, "lr": self.cfg.capture_lr}
        return {"server_round": server_round, "fedsgd": False, "batch_size": self.cfg.batch_size,
                "lr": self.cfg.lr, "local_epochs": self.cfg.local_epochs}

    def configure_fit(self, server_round, parameters, client_manager):
        self._round_start = time.time()
        if server_round == self.cfg.capture_round:
            self._sent_in_capture_round = parameters
        clients = client_manager.sample(num_clients=self.cfg.num_clients, min_num_clients=self.cfg.num_clients)
        config = self._fit_config(server_round)
        return [(client, FitIns(parameters, config)) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        if failures:
            raise RuntimeError(f"{len(failures)} client(s) failed in round {server_round}: {failures[0]!r}")
        upload_mb = np.mean([sum(len(t) for t in res.parameters.tensors) for _, res in results]) / 2 ** 20

        if server_round == self.cfg.capture_round:
            self._save_observation(server_round, results)

        weights_results = [(security.parameters_to_weights(res.parameters, self.public_ctx), res.num_examples)
                           for _, res in results]
        aggregated = security.weights_to_parameters(security.aggregate_custom(weights_results))

        self.history.append({"round": server_round, "upload_mb_per_client": round(float(upload_mb), 2),
                             "fedsgd": server_round == self.cfg.capture_round})
        if server_round == self.cfg.rounds:
            with open(os.path.join(self.cfg.out_dir, "final_global_parameters.pkl"), "wb") as f:
                pickle.dump({"tensor_type": aggregated.tensor_type, "tensors": aggregated.tensors}, f)
        return aggregated, {}

    def _save_observation(self, server_round, results):
        """Everything an honest-but-curious server holds after receiving the round's updates."""
        sent = self._sent_in_capture_round
        observation = {
            "round": server_round,
            "fit_config": self._fit_config(server_round),
            "tensor_type": sent.tensor_type,
            "global_tensors": sent.tensors,
            "updates": {proxy.cid: {"tensors": res.parameters.tensors, "tensor_type": res.parameters.tensor_type,
                                    "num_examples": res.num_examples}
                        for proxy, res in results},
        }
        with open(os.path.join(self.cfg.out_dir, "server_observation.pkl"), "wb") as f:
            pickle.dump(observation, f)

    def configure_evaluate(self, server_round, parameters, client_manager):
        clients = client_manager.sample(num_clients=self.cfg.num_clients, min_num_clients=self.cfg.num_clients)
        return [(client, EvaluateIns(parameters, {})) for client in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        if failures:
            raise RuntimeError(f"{len(failures)} client(s) failed evaluation in round {server_round}: {failures[0]!r}")
        n = sum(res.num_examples for _, res in results)
        loss = sum(res.loss * res.num_examples for _, res in results) / n
        accuracy = sum(res.metrics["accuracy"] * res.num_examples for _, res in results) / n
        entry = self.history[-1]
        entry.update(val_loss=round(loss, 4), val_accuracy=round(accuracy, 2),
                     round_seconds=round(time.time() - self._round_start, 1))
        print(f"  round {server_round:2d} | val acc {accuracy:5.1f}% | val loss {loss:.3f} | "
              f"upload {entry['upload_mb_per_client']:.2f} MB/client | {entry['round_seconds']:.0f}s"
              + ("  <- FedSGD capture round" if entry["fedsgd"] else ""))
        return loss, {"accuracy": accuracy}

    def evaluate(self, server_round, parameters):
        return None  # the server holds no data (and with FHE it cannot read the model)


# ------------------------------------------------------------------ driver

def decrypt_final_model(cfg: FLConfig) -> MRINet:
    """What a client does after training: load (and, with FHE, decrypt) the final global model."""
    with open(os.path.join(cfg.out_dir, "final_global_parameters.pkl"), "rb") as f:
        saved = pickle.load(f)
    parameters = Parameters(tensors=saved["tensors"], tensor_type=saved["tensor_type"])
    ctx = load_context(ensure_keys(cfg.keys_dir)[0]) if parameters.tensor_type == security.TENSOR_TYPE_CKKS else None
    model = MRINet(num_classes=4, img_size=cfg.img_size)
    set_parameters(model, security.parameters_to_weights(parameters, ctx), ctx, save_path=None)
    return model.eval()


def test_accuracy(model, cfg: FLConfig) -> float:
    arrays = data_setup.load_mri_arrays(cfg.data_path, cfg.img_size)
    test_ds = TensorDataset(data_setup.to_model_input(arrays["x_test"]), torch.as_tensor(arrays["y_test"]))
    _, accuracy, *_ = engine.test(model, DataLoader(test_ds, batch_size=256), torch.nn.CrossEntropyLoss(), "cpu")
    return float(accuracy)


def run_federated(cfg: FLConfig) -> dict:
    """Run the Flower simulation; returns a summary and writes everything under cfg.out_dir."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    data_setup.load_mri_arrays(cfg.data_path, cfg.img_size)   # build the cache before the actors start
    public_ctx = load_context(ensure_keys(cfg.keys_dir)[1]) if cfg.he else None

    os.environ["PYTHONPATH"] = SRC_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")  # for Ray workers
    os.environ.setdefault("FLWR_TELEMETRY_ENABLED", "0")
    flwr_logger.setLevel(logging.WARNING)

    strategy = CapturingFedAvg(cfg, public_ctx)
    print(f"Federated training: {'FHE (CKKS)' if cfg.he else 'plaintext'} | {cfg.num_clients} clients | "
          f"{cfg.partition} | {cfg.rounds} rounds")
    start = time.time()
    fl.simulation.start_simulation(
        client_fn=functools.partial(_client_fn, cfg=cfg),
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.rounds),
        strategy=strategy,
        client_resources={"num_cpus": cfg.cpus_per_client},
        ray_init_args={"include_dashboard": False, "ignore_reinit_error": True,
                       "num_cpus": cfg.cpus_per_client * cfg.num_clients, "logging_level": logging.ERROR},
    )
    model = decrypt_final_model(cfg)
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "final_model.pt"))
    summary = {"config": asdict(cfg), "test_accuracy": round(test_accuracy(model, cfg), 2),
               "training_seconds": round(time.time() - start, 1), "rounds": strategy.history}
    with open(os.path.join(cfg.out_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Test accuracy of the final global model: {summary['test_accuracy']:.1f}%")
    return summary
