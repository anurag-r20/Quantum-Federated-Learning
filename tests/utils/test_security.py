"""Unit tests for src/utils/security.py: CKKS encryption of model weights and FedAvg over ciphertexts.

One test per promised behaviour, named test__<unit>__<scenario>__<expected outcome>, laid out as
Arrange / Act / Assert. Expected values come from hand calculations, identities (decrypt o encrypt = id),
an independent numpy implementation, or the security property itself; never from the code under test.
"""

from __future__ import annotations

import numpy as np
import pytest
import tenseal as ts
import torch

from utils import security

# CKKS at scale 2^40 decrypts |x| <= 10 to about 1e-7; 1e-4 leaves a wide margin without hiding real errors.
CKKS_ATOL = 1e-4


@pytest.mark.parametrize(
    "shape",
    [(1,), (16,), (3, 3), (4096,), (4097,), (2, 3, 1)],
    ids=["one-value", "vector", "matrix", "exactly-one-ciphertext-of-slots", "slots-plus-one", "trailing-unit-axis"],
)
def test__encrypt_array__given_any_shape__decrypts_to_the_original_values(ckks_secret_context, shape) -> None:
    """Identity oracle: decrypt(encrypt(x)) == x up to CKKS noise, with the original shape restored."""
    # Arrange
    values = np.random.default_rng(0).normal(0, 1, size=shape)

    # Act
    decrypted = security.decrypt_array(security.encrypt_array(ckks_secret_context, values), shape)

    # Assert
    assert decrypted.shape == shape
    np.testing.assert_allclose(decrypted, values, atol=CKKS_ATOL)


@pytest.mark.parametrize(
    "n_values, expected_ciphertexts",
    [(1, 2), (4096, 2), (8192, 2), (8193, 3), (40960, 10)],
    ids=["one-value", "one-slot-block", "two-slot-blocks", "just-over-two", "ten-slot-blocks"],
)
def test__encrypt_array__given_n_values__uses_max_of_two_and_ceil_n_over_4096_ciphertexts(
    ckks_secret_context, n_values, expected_ciphertexts
) -> None:
    """Known oracle: N = 8192 gives 8192 / 2 = 4096 slots per ciphertext; at least 2 ciphertexts are used
    because TenSEAL decrypts a batched (4096, 1) tensor as its first value repeated."""
    # Act
    encrypted = security.encrypt_array(ckks_secret_context, np.zeros(n_values))

    # Assert
    assert encrypted.shape == [expected_ciphertexts]


def test__decrypt_array__given_the_servers_public_context__raises_value_error(ckks_public_context) -> None:
    """Security property: the public context can encrypt, but holds no secret key to decrypt with."""
    # Arrange
    encrypted = security.encrypt_array(ckks_public_context, np.ones(5))

    # Act
    with pytest.raises(Exception) as error:
        security.decrypt_array(encrypted, (5,))

    # Assert
    assert isinstance(error.value, ValueError)


def test__aggregate_custom__given_plaintext_updates__returns_the_example_weighted_mean() -> None:
    """Known oracle: A = [1, 2] with 1 example, B = [4, 8] with 3 examples:
    ((1*1 + 4*3) / 4, (2*1 + 8*3) / 4) = (3.25, 6.5)."""
    # Arrange
    results = [([np.array([1.0, 2.0])], 1), ([np.array([4.0, 8.0])], 3)]

    # Act
    (aggregated,) = security.aggregate_custom(results)

    # Assert
    assert aggregated.tolist() == pytest.approx([3.25, 6.5])


def test__aggregate_custom__given_encrypted_updates__decrypts_to_the_plaintext_weighted_mean(
    ckks_secret_context, ckks_public_context
) -> None:
    """Differential oracle: FedAvg computed by the server on ciphertexts equals numpy's weighted average."""
    # Arrange
    updates = np.random.default_rng(1).normal(0, 0.1, size=(3, 50))
    counts = [10, 30, 60]
    encrypted = [([security.encrypt_array(ckks_public_context, u)], n) for u, n in zip(updates, counts)]

    # Act
    (aggregated,) = security.aggregate_custom(encrypted)

    # Assert
    client_view = ts.ckks_tensor_from(ckks_secret_context, aggregated.serialize())
    np.testing.assert_allclose(security.decrypt_array(client_view, (50,)),
                               np.average(updates, axis=0, weights=counts), atol=CKKS_ATOL)


def test__crypte__given_a_state_dict_without_an_fc3_layer__encrypts_every_tensor(ckks_public_context) -> None:
    """Contract: every tensor is encrypted. Regression for the original code, which encrypted only a layer
    named 'fc3.weight' and therefore nothing at all for the QFL models."""
    # Arrange
    state = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3), torch.nn.Linear(4, 2)).state_dict()

    # Act
    layers = security.crypte(state, ckks_public_context)

    # Assert
    assert [layer.get_name() for layer in layers] == list(state.keys())
    assert all(isinstance(layer.get_weight(), ts.tensors.CKKSTensor) for layer in layers)


def test__crypted_layer_decrypt__given_the_layer_shape__restores_the_original_tensor(ckks_secret_context) -> None:
    """Identity oracle: a 2x3 weight encrypted by CryptedLayer decrypts back to the same 2x3 values."""
    # Arrange
    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    layer = security.CryptedLayer("w", weight, ckks_secret_context)

    # Act
    decrypted = layer.decrypt(ckks_secret_context.secret_key(), shape=(2, 3))

    # Assert
    np.testing.assert_allclose(decrypted, weight.numpy(), atol=CKKS_ATOL)


def test__layer_len__given_a_2x3_array__returns_6() -> None:
    """Known oracle: 2 * 3 = 6 weights. (The original __len__ called array.shape() and raised TypeError.)"""
    # Act
    size = len(security.Layer("w", np.zeros((2, 3))))

    # Assert
    assert size == 6


def test__weights_to_parameters__given_numpy_weights__round_trips_values_dtypes_and_shapes() -> None:
    """Identity oracle: parameters_to_weights(weights_to_parameters(w)) == w, marked as numpy."""
    # Arrange
    weights = [np.arange(6, dtype=np.float32).reshape(2, 3), np.array([1.5], dtype=np.float32)]

    # Act
    parameters = security.weights_to_parameters(weights)
    restored = security.parameters_to_weights(parameters)

    # Assert
    assert parameters.tensor_type == security.TENSOR_TYPE_NUMPY
    assert all(r.dtype == w.dtype and np.array_equal(r, w) for r, w in zip(restored, weights))


def test__parameters_to_weights__given_encrypted_parameters_and_no_context__raises_value_error(
    ckks_public_context,
) -> None:
    # Arrange
    parameters = security.weights_to_parameters([security.encrypt_array(ckks_public_context, np.ones(3))])

    # Act
    with pytest.raises(Exception) as error:
        security.parameters_to_weights(parameters)

    # Assert
    assert parameters.tensor_type == security.TENSOR_TYPE_CKKS
    assert isinstance(error.value, ValueError)


def test__deserialized_layer__given_numpy_bytes__returns_a_plaintext_layer(ckks_secret_context) -> None:
    # Arrange
    serialized = security.weights_to_parameters([np.array([1.0, 2.0])]).tensors[0]

    # Act
    layer = security.deserialized_layer("b", serialized, ckks_secret_context)

    # Assert
    assert type(layer) is security.Layer
    assert layer.decrypt().tolist() == pytest.approx([1.0, 2.0])


def test__deserialized_layer__given_ciphertext_bytes__returns_a_crypted_layer(ckks_secret_context) -> None:
    # Arrange
    serialized = security.encrypt_array(ckks_secret_context, np.array([1.0, 2.0])).serialize()

    # Act
    layer = security.deserialized_layer("b", serialized, ckks_secret_context)

    # Assert
    assert type(layer) is security.CryptedLayer
    np.testing.assert_allclose(layer.decrypt(shape=(2,)), [1.0, 2.0], atol=CKKS_ATOL)


def test__read_query__given_a_written_query__returns_the_items_and_the_context_separately(tmp_path) -> None:
    # Arrange
    path = tmp_path / "query.pkl"
    security.write_query(path, {"contexte": b"serialized-context", "layer": 3})

    # Act
    query, contexte = security.read_query(path)

    # Assert
    assert query == {"layer": 3}
    assert contexte == b"serialized-context"


def test__read_query__given_a_missing_file__raises_file_not_found_error(tmp_path) -> None:
    # Act
    with pytest.raises(Exception) as error:
        security.read_query(tmp_path / "missing.pkl")

    # Assert
    assert isinstance(error.value, FileNotFoundError)


def test__log_metric__given_no_folder__writes_nothing(tmp_path, monkeypatch) -> None:
    # Arrange
    monkeypatch.chdir(tmp_path)

    # Act
    security.log_metric("message", None)

    # Assert
    assert list(tmp_path.iterdir()) == []


def test__log_metric__given_a_folder_twice__appends_one_timestamped_line_per_call(tmp_path) -> None:
    # Act
    security.log_metric("first", str(tmp_path))
    security.log_metric("second", str(tmp_path))

    # Assert
    lines = (tmp_path / "timing_log.txt").read_text().splitlines()
    assert [line.split("] ", 1)[1] for line in lines] == ["first", "second"]
    assert all(line.startswith("[") for line in lines)


def test__bytes_to_ndarray_ckks__given_numpy_bytes__returns_the_original_array() -> None:
    """Identity oracle: NumPy payloads keep Flower's behaviour."""
    # Arrange
    array = np.array([[1.0, -2.5]], dtype=np.float32)

    # Act
    restored = security.bytes_to_ndarray_ckks(security.ndarray_to_bytes_ckks(array))

    # Assert
    assert restored.dtype == array.dtype and np.array_equal(restored, array)


def test__bytes_to_ndarray_ckks__given_ciphertext_bytes__returns_them_unchanged(ckks_public_context) -> None:
    """Contract: the server-side converter never tries to read a ciphertext; it hands the bytes on."""
    # Arrange
    serialized = security.encrypt_array(ckks_public_context, np.ones(4)).serialize()

    # Act
    restored = security.bytes_to_ndarray_ckks(serialized)

    # Assert
    assert restored == serialized


def test__enable_ckks_in_flower__given_a_ckks_tensor__lets_flower_serialize_it(ckks_public_context) -> None:
    """Regression: Flower's default converter calls np.save on the CKKS tensor and raises (the FHE notebooks'
    clients crashed this way in Ray workers once encryption was real)."""
    # Arrange
    import flwr.common as flwr_common

    encrypted = security.encrypt_array(ckks_public_context, np.ones(4))
    security.enable_ckks_in_flower()

    # Act
    parameters = flwr_common.ndarrays_to_parameters([np.zeros(2), encrypted])

    # Assert
    assert parameters.tensors[1] == encrypted.serialize()
