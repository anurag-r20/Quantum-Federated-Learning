"""Unit tests for src/federated.py: keys, client data splits, the Flower client and the capturing strategy.

Everything runs in-process on a synthetic 16x16 MRI cache (no Ray, no real data). Oracles: SGD's definition
(w_client = w_global - lr * grad, with grad from autograd), numpy weighted means, MRINet's seeded
initialization, and the encrypt/decrypt identity.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import tenseal as ts
import torch
import torch.nn.functional as F
from flwr.common import Code, EvaluateRes, FitIns, FitRes, Status
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy

import federated
from models import MRINet
from utils import security

OK = Status(code=Code.OK, message="")


class OfflineClientProxy(ClientProxy):
    """Stands in for a remote client at the network boundary; the strategy never calls it in these tests."""

    def get_properties(self, ins, timeout):
        raise NotImplementedError

    def get_parameters(self, ins, timeout):
        raise NotImplementedError

    def fit(self, ins, timeout):
        raise NotImplementedError

    def evaluate(self, ins, timeout):
        raise NotImplementedError

    def reconnect(self, ins, timeout):
        raise NotImplementedError


def config(tmp_path, data_path=None, **overrides) -> federated.FLConfig:
    values = dict(data_path=str(data_path or tmp_path / "MRI"), img_size=16, num_clients=3, rounds=5,
                  capture_round=2, out_dir=str(tmp_path / "out"), keys_dir=str(tmp_path / "keys"),
                  cpus_per_client=torch.get_num_threads())
    values.update(overrides)
    (tmp_path / "out").mkdir(exist_ok=True)
    return federated.FLConfig(**values)


def seeded_weights(seed: int = 0) -> list[np.ndarray]:
    """Oracle for the initial model: MRINet(16px) right after torch.manual_seed(seed)."""
    torch.manual_seed(seed)
    return [v.numpy() for v in MRINet(num_classes=4, img_size=16).state_dict().values()]


def client_manager(n: int = 3) -> SimpleClientManager:
    manager = SimpleClientManager()
    for cid in range(n):
        manager.register(OfflineClientProxy(str(cid)))
    return manager


def test__ensure_keys__given_an_empty_folder__writes_a_secret_client_key_and_a_public_server_key(tmp_path) -> None:
    # Act
    secret_path, public_path = federated.ensure_keys(str(tmp_path / "keys"))

    # Assert
    assert federated.load_context(secret_path).is_private()
    assert not federated.load_context(public_path).is_private()


def test__ensure_keys__given_existing_keys__keeps_them(tmp_path) -> None:
    """Relation: a second call must not replace the key pair (clients would lose access to old ciphertexts)."""
    # Arrange
    secret_path, _ = federated.ensure_keys(str(tmp_path / "keys"))
    before = open(secret_path, "rb").read()

    # Act
    federated.ensure_keys(str(tmp_path / "keys"))

    # Assert
    assert open(secret_path, "rb").read() == before


def test__client_dataset__given_a_client__returns_disjoint_train_and_validation_sets_of_the_right_size(
    tmp_path, make_mri_cache
) -> None:
    """Known oracle: 48 training images, 3 IID clients of 16; validation = int(16 * 0.1) = 1, training = 15."""
    # Arrange
    cfg = config(tmp_path, data_path=make_mri_cache(tmp_path))

    # Act
    train, val = federated.client_dataset(cfg, 0, "train"), federated.client_dataset(cfg, 0, "val")

    # Assert
    assert (len(train), len(val)) == (15, 1)
    train_rows = {tuple(x.flatten().tolist()) for x in train.tensors[0]}
    assert all(tuple(x.flatten().tolist()) not in train_rows for x in val.tensors[0])


@pytest.mark.parametrize("server_round, fedsgd", [(2, True), (3, False)], ids=["capture-round", "fedavg-round"])
def test__configure_fit__given_a_round__asks_every_client_for_fedsgd_only_in_the_capture_round(
    tmp_path, server_round, fedsgd
) -> None:
    # Arrange
    cfg = config(tmp_path, capture_batch_size=1, capture_lr=0.01, local_epochs=2, lr=1e-3)
    strategy = federated.CapturingFedAvg(cfg)
    parameters = security.weights_to_parameters(seeded_weights())

    # Act
    instructions = strategy.configure_fit(server_round, parameters, client_manager())

    # Assert
    configs = [fit_ins.config for _, fit_ins in instructions]
    assert len(configs) == 3 and all(c["fedsgd"] is fedsgd for c in configs)
    expected = {"batch_size": 1, "lr": 0.01} if fedsgd else {"batch_size": 32, "lr": 1e-3, "local_epochs": 2}
    assert all(expected.items() <= c.items() for c in configs)


def test__initialize_parameters__given_plaintext__returns_the_seeded_mrinet_weights(tmp_path) -> None:
    # Act
    parameters = federated.CapturingFedAvg(config(tmp_path)).initialize_parameters(client_manager())

    # Assert
    assert parameters.tensor_type == security.TENSOR_TYPE_NUMPY
    assert all(np.array_equal(a, b) for a, b in zip(security.parameters_to_weights(parameters), seeded_weights()))


def test__initialize_parameters__given_fhe__encrypts_the_seeded_weights_with_the_public_key(
    tmp_path, ckks_secret_context, ckks_public_context
) -> None:
    """Identity oracle: the server encrypts with the public key; the client's secret key recovers the weights."""
    # Act
    parameters = federated.CapturingFedAvg(config(tmp_path, he=True), ckks_public_context).initialize_parameters(
        client_manager())

    # Assert
    assert parameters.tensor_type == security.TENSOR_TYPE_CKKS
    for tensor, expected in zip(security.parameters_to_weights(parameters, ckks_secret_context), seeded_weights()):
        np.testing.assert_allclose(security.decrypt_array(tensor, expected.shape), expected, atol=1e-4)


def test__aggregate_fit__given_two_plaintext_updates__returns_their_example_weighted_mean(tmp_path) -> None:
    """Oracle: numpy's weighted average with weights 1 and 3."""
    # Arrange
    strategy = federated.CapturingFedAvg(config(tmp_path))
    a, b = seeded_weights(1), seeded_weights(2)
    results = [(OfflineClientProxy("0"), FitRes(OK, security.weights_to_parameters(a), 1, {})),
               (OfflineClientProxy("1"), FitRes(OK, security.weights_to_parameters(b), 3, {}))]

    # Act
    aggregated, _ = strategy.aggregate_fit(3, results, [])

    # Assert
    for got, wa, wb in zip(security.parameters_to_weights(aggregated), a, b):
        np.testing.assert_allclose(got, (wa + 3 * wb) / 4, rtol=1e-6, atol=1e-7)


def test__aggregate_fit__given_the_capture_round__records_what_the_server_received(tmp_path) -> None:
    """Contract: the observation holds the global weights sent and every client's update, byte for byte."""
    # Arrange
    cfg = config(tmp_path)
    strategy = federated.CapturingFedAvg(cfg)
    sent = security.weights_to_parameters(seeded_weights(0))
    strategy.configure_fit(cfg.capture_round, sent, client_manager())
    update = security.weights_to_parameters(seeded_weights(1))
    results = [(OfflineClientProxy("2"), FitRes(OK, update, 1, {}))]

    # Act
    strategy.aggregate_fit(cfg.capture_round, results, [])

    # Assert
    with open(tmp_path / "out" / "server_observation.pkl", "rb") as f:
        observation = pickle.load(f)
    assert observation["global_tensors"] == sent.tensors
    assert observation["updates"]["2"]["tensors"] == update.tensors
    assert observation["fit_config"]["fedsgd"] is True


def test__aggregate_fit__given_a_failed_client__raises_runtime_error(tmp_path) -> None:
    # Arrange
    strategy = federated.CapturingFedAvg(config(tmp_path))

    # Act
    with pytest.raises(Exception) as error:
        strategy.aggregate_fit(3, [], [RuntimeError("client crashed")])

    # Assert
    assert isinstance(error.value, RuntimeError)


def test__aggregate_evaluate__given_two_clients__returns_the_example_weighted_accuracy(tmp_path) -> None:
    """Known oracle: 50% on 1 example and 100% on 3 examples -> (50 + 300) / 4 = 87.5%."""
    # Arrange
    strategy = federated.CapturingFedAvg(config(tmp_path))
    strategy.configure_fit(3, security.weights_to_parameters(seeded_weights()), client_manager())
    update = security.weights_to_parameters(seeded_weights())
    strategy.aggregate_fit(3, [(OfflineClientProxy("0"), FitRes(OK, update, 1, {}))], [])
    results = [(OfflineClientProxy("0"), EvaluateRes(OK, 0.4, 1, {"accuracy": 50.0})),
               (OfflineClientProxy("1"), EvaluateRes(OK, 0.2, 3, {"accuracy": 100.0}))]

    # Act
    loss, metrics = strategy.aggregate_evaluate(3, results, [])

    # Assert
    assert metrics["accuracy"] == pytest.approx(87.5)
    assert loss == pytest.approx((0.4 + 3 * 0.2) / 4)


def test__mri_client_fit__given_a_fedsgd_round__returns_global_weights_minus_lr_times_the_gradient(
    tmp_path, make_mri_cache
) -> None:
    """Oracle: SGD's definition, with the gradient recomputed by autograd on the batch the client saved."""
    # Arrange
    cfg = config(tmp_path, data_path=make_mri_cache(tmp_path))
    client = federated.MRIClient("0", cfg)
    global_weights = seeded_weights()
    ins = FitIns(security.weights_to_parameters(global_weights),
                 {"server_round": 2, "fedsgd": True, "batch_size": 2, "lr": 0.05})

    # Act
    result = client.fit(ins)

    # Assert
    truth = torch.load(tmp_path / "out" / "gia_ground_truth" / "client0.pt")
    model = MRINet(num_classes=4, img_size=16)
    model.load_state_dict({k: torch.tensor(w) for k, w in zip(model.state_dict().keys(), global_weights)})
    grads = torch.autograd.grad(F.cross_entropy(model(truth["images"]), truth["labels"]), list(model.parameters()))
    assert result.num_examples == 2 and len(truth["labels"]) == 2
    for got, w, g in zip(security.parameters_to_weights(result.parameters), global_weights, grads):
        np.testing.assert_allclose(got, w - 0.05 * g.numpy(), rtol=1e-5, atol=1e-6)


def test__mri_client_fit__given_fhe__returns_ciphertexts_that_decrypt_to_the_plaintext_update(
    tmp_path, make_mri_cache
) -> None:
    """Differential oracle: the same FedSGD step with and without encryption gives the same weights."""
    # Arrange
    data_path = make_mri_cache(tmp_path)
    ins = FitIns(security.weights_to_parameters(seeded_weights()),
                 {"server_round": 2, "fedsgd": True, "batch_size": 2, "lr": 0.05})
    plain = federated.MRIClient("1", config(tmp_path, data_path=data_path)).fit(ins)
    cfg = config(tmp_path, data_path=data_path, he=True)
    secret = federated.load_context(federated.ensure_keys(cfg.keys_dir)[0])

    # Act
    encrypted = federated.MRIClient("1", cfg).fit(ins)

    # Assert
    assert encrypted.parameters.tensor_type == security.TENSOR_TYPE_CKKS
    expected = security.parameters_to_weights(plain.parameters)
    for tensor, want in zip(security.parameters_to_weights(encrypted.parameters, secret), expected):
        np.testing.assert_allclose(security.decrypt_array(tensor, want.shape), want, atol=1e-4)


def test__decrypt_final_model__given_saved_plaintext_parameters__loads_them_into_an_mrinet(tmp_path) -> None:
    # Arrange
    cfg = config(tmp_path)
    weights = seeded_weights(3)
    saved = security.weights_to_parameters(weights)
    with open(tmp_path / "out" / "final_global_parameters.pkl", "wb") as f:
        pickle.dump({"tensor_type": saved.tensor_type, "tensors": saved.tensors}, f)

    # Act
    model = federated.decrypt_final_model(cfg)

    # Assert
    assert all(np.array_equal(v.numpy(), w) for v, w in zip(model.state_dict().values(), weights))
