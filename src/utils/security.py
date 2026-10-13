"""
CKKS homomorphic encryption (TenSEAL) for federated learning model updates.

Every model tensor is flattened, zero-padded and packed into the slots of batched
CKKS ciphertexts, so a whole state_dict can be encrypted (one ciphertext per
SLOTS weights). The server only ever holds the public context: it can encrypt and
add/scale ciphertexts (FedAvg), but it cannot decrypt anything.

Adapted from data-science-lover/flower-homomorphic_encryption (Apache-2.0), via QFL-MLNCP.
Changed: every tensor is encrypted (the original encrypted only a layer named 'fc3.weight', which
none of the QFL models have), batched CKKS packing, shape-restoring decryption, single-multiply FedAvg.
"""
from typing import List, Optional, Sequence, Tuple
from functools import reduce
from io import BytesIO
import math
import operator
import os
import pickle
import time

import numpy as np
import tenseal as ts
from flwr.common import NDArrays, Parameters

# CKKS parameters: N=8192 gives 4096 slots per ciphertext and ~128-bit security
# with these moduli; two 40-bit levels are enough for one plaintext-scalar multiply.
POLY_MODULUS_DEGREE = 8192
COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
GLOBAL_SCALE = 2 ** 40
SLOTS = POLY_MODULUS_DEGREE // 2

TENSOR_TYPE_CKKS = "tenseal.ckks"
TENSOR_TYPE_NUMPY = "numpy.ndarray"


def log_metric(message: str, save_results_path: Optional[str] = 'results/FL/'):
    """Appends a timestamped message to <save_results_path>/timing_log.txt (no-op if path is None)."""
    if save_results_path is None:
        return
    os.makedirs(save_results_path, exist_ok=True)
    log_file_path = os.path.join(save_results_path, 'timing_log.txt')
    try:
        with open(log_file_path, 'a') as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        print(f"CRITICAL: Log file write failed. Error: {e} - Message: {message}")


# /////////////////////// Homomorphic Encryption \\\\\\\\\\\\\\\\\\\\\\\\\\

def context():
    """
    Create a CKKS context holding a fresh key pair (secret, public and relinearization keys).

    :return: the private TenSEAL context; share `ctx.serialize(save_secret_key=False)` with the server
    """
    cont = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=POLY_MODULUS_DEGREE,
                      coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES)
    cont.global_scale = GLOBAL_SCALE
    return cont


def encrypt_array(ctx, array) -> ts.tensors.CKKSTensor:
    """
    Encrypt an array of any shape into a batched CKKS tensor.

    The flattened array is zero-padded and reshaped to (SLOTS, n_ct): each of the n_ct
    ciphertexts packs SLOTS values. At least 2 ciphertexts are used because TenSEAL
    decrypts a batched (SLOTS, 1) tensor incorrectly (every slot returns the first value).
    """
    flat = np.asarray(array, dtype=np.float64).ravel()
    n_ct = max(2, math.ceil(flat.size / SLOTS))
    packed = np.zeros(SLOTS * n_ct)
    packed[:flat.size] = flat
    return ts.ckks_tensor(ctx, packed.reshape(SLOTS, n_ct), batch=True)


def decrypt_array(enc, shape: Sequence[int], secret_key=None) -> np.ndarray:
    """Decrypt a tensor produced by `encrypt_array` and restore its original shape."""
    plain = enc.decrypt(secret_key) if secret_key is not None else enc.decrypt()
    flat = np.asarray(plain.tolist(), dtype=np.float64).ravel()
    n = int(np.prod(shape)) if len(shape) else 1
    return flat[:n].reshape(shape).astype(np.float32)


class Layer:
    """
    This class is used to represent the (plaintext) weights of a layer of a neural network.

    :param name_layer: the name of the layer
    :param weight: the weights of the layer
    """
    def __init__(self, name_layer, weight):
        self.name = name_layer
        self.weight_array = weight

    def get_name(self):
        return self.name

    def get_weight(self):
        return self.weight_array

    def __add__(self, other):
        weights = other.get_weight() if isinstance(other, Layer) else other
        return Layer(self.name, self.weight_array + weights)

    def __sub__(self, other):
        weights = other.get_weight() if isinstance(other, Layer) else other
        return Layer(self.name, self.weight_array - weights)

    def __mul__(self, other):
        weights = other.get_weight() if isinstance(other, Layer) else other
        return Layer(self.name, self.weight_array * weights)

    def __truediv__(self, other):
        weights = other.get_weight() if isinstance(other, Layer) else other
        return Layer(self.name, self.weight_array * (1 / weights))

    def __len__(self):
        return int(np.prod(np.shape(self.weight_array)))

    def shape(self):
        return np.shape(self.weight_array)

    def sum(self, axis=0):
        return Layer(f"sum_{self.name}", self.weight_array.sum(axis=axis))

    def mean(self, axis=0):
        weights = self.weight_array.sum(axis=axis) * (1 / np.shape(self.weight_array)[axis])
        return Layer(f"sum_{self.name}", weights)

    def decrypt(self, sk=None, shape=None):
        """
        Return the weights as a float32 array (nothing to decrypt; same interface as CryptedLayer).

        :param sk: unused
        :param shape: optional shape to reshape the weights to
        """
        weights = np.asarray(self.weight_array, dtype=np.float32)
        return weights.reshape(shape) if shape is not None else weights

    def serialize(self):
        return {self.name: self.weight_array}


class CryptedLayer(Layer):
    """
    This class is used to represent the encrypted weights of a layer of a neural network.

    :param name_layer: the name of the layer
    :param weight: a CKKSTensor, its serialized bytes, or a plaintext tensor/array to encrypt
    :param contexte: the context of the encryption (needed to encrypt plaintext weights)
    """
    def __init__(self, name_layer, weight, contexte=None):
        super(CryptedLayer, self).__init__(name_layer, weight)
        if isinstance(weight, (ts.tensors.CKKSTensor, bytes)):
            self.weight_array = weight
        else:
            if hasattr(weight, "detach"):
                weight = weight.detach().cpu().numpy()
            self.weight_array = encrypt_array(contexte, weight)

    def __add__(self, other):
        weights = other.get_weight() if isinstance(other, CryptedLayer) else other
        return CryptedLayer(self.name, self.weight_array + weights)

    def __sub__(self, other):
        weights = other.get_weight() if isinstance(other, CryptedLayer) else other
        return CryptedLayer(self.name, self.weight_array - weights)

    def __mul__(self, other):
        weights = other.get_weight() if isinstance(other, CryptedLayer) else other
        return CryptedLayer(self.name, self.weight_array * weights)

    def __truediv__(self, other):
        # Only division by a plaintext number is possible (multiplication by its inverse)
        return CryptedLayer(self.name, self.weight_array * (1 / other))

    def shape(self):
        return self.weight_array.shape

    def decrypt(self, sk=None, shape=None):
        """
        Decrypt the weights with the secret key.

        :param sk: the secret key (default: the one held by the tensor's context, if any)
        :param shape: original shape of the layer; without it the padded (SLOTS, n_ct) layout is returned
        """
        if shape is not None:
            return decrypt_array(self.weight_array, shape, sk)
        plain = self.weight_array.decrypt(sk) if sk is not None else self.weight_array.decrypt()
        return np.asarray(plain.tolist(), dtype=np.float32)

    def serialize(self):
        return {self.name: self.weight_array.serialize()}


def crypte(client_w, context_c):
    """
    Encrypt every tensor of a state_dict.

    :param client_w: dictionary name -> weights of a neural network
    :param context_c: the context of the encryption (the public context is enough)
    :return: a list of CryptedLayer objects
    """
    return [CryptedLayer(name_layer, weight_array, context_c) for name_layer, weight_array in client_w.items()]


def read_query(file_path):
    """
    Read a pickled query {"contexte": <serialized context>, ...}.

    :param file_path: the path of the file to read
    :return: the remaining query items and the serialized context
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    with open(file_path, 'rb') as file:
        query = pickle.load(file)
    contexte = query.pop("contexte")
    return query, contexte


def write_query(file_path, client_query):
    """
    Pickle a query to a file.

    :param file_path: the path of the file to write
    :param client_query: the query to write
    """
    with open(file_path, 'wb') as file:
        pickle.dump(client_query, file)


def deserialized_layer(name_layer, weight_array, ctx):
    """
    Turn serialized/encrypted/plain weights into a Layer or CryptedLayer.

    :param name_layer: the name of the layer
    :param weight_array: CKKSTensor, serialized bytes (ciphertext or np.save output) or array
    :param ctx: the context (if the layer is encrypted)
    """
    if isinstance(weight_array, bytes):
        try:
            return CryptedLayer(name_layer, ts.ckks_tensor_from(ctx, weight_array), ctx)
        except Exception:
            return Layer(name_layer, np.load(BytesIO(weight_array), allow_pickle=False))
    if isinstance(weight_array, ts.tensors.CKKSTensor):
        return CryptedLayer(name_layer, weight_array, ctx)
    return Layer(name_layer, weight_array)


def deserialized_model(client_query, ctx):
    """Deserialize every layer of a {name: weights} query."""
    return [deserialized_layer(name_layer, weight_array, ctx) for name_layer, weight_array in client_query.items()]


# //////////////////////// Flower Parameters <-> weights \\\\\\\\\\\\\\\\\\\\\\\\

def weights_to_parameters(weights) -> Parameters:
    """Serialize a list of CKKSTensors or arrays into Flower Parameters (tensor_type tells which)."""
    if weights and isinstance(weights[0], ts.tensors.CKKSTensor):
        return Parameters(tensors=[w.serialize() for w in weights], tensor_type=TENSOR_TYPE_CKKS)
    tensors = []
    for w in weights:
        buffer = BytesIO()
        np.save(buffer, np.asarray(w), allow_pickle=False)
        tensors.append(buffer.getvalue())
    return Parameters(tensors=tensors, tensor_type=TENSOR_TYPE_NUMPY)


def parameters_to_weights(parameters: Parameters, ctx=None) -> list:
    """Deserialize Flower Parameters into CKKSTensors (needs a context) or numpy arrays."""
    if parameters.tensor_type == TENSOR_TYPE_CKKS:
        if ctx is None:
            raise ValueError("A TenSEAL context is required to load encrypted parameters")
        return [ts.ckks_tensor_from(ctx, tensor) for tensor in parameters.tensors]
    return [np.load(BytesIO(tensor), allow_pickle=False) for tensor in parameters.tensors]


def ndarray_to_bytes_ckks(ndarray) -> bytes:
    """Flower's ndarray_to_bytes, extended: CKKS tensors are serialized, bytes pass through unchanged."""
    if isinstance(ndarray, ts.tensors.CKKSTensor):
        return ndarray.serialize()
    if isinstance(ndarray, bytes):
        return ndarray
    if hasattr(ndarray, "detach"):
        ndarray = ndarray.detach().cpu().numpy()
    buffer = BytesIO()
    np.save(buffer, ndarray, allow_pickle=False)
    return buffer.getvalue()


def bytes_to_ndarray_ckks(tensor: bytes):
    """Flower's bytes_to_ndarray, extended: a CKKS ciphertext is returned as bytes, for a key holder to decrypt."""
    try:
        return np.load(BytesIO(tensor), allow_pickle=False)
    except ValueError:
        return tensor


def enable_ckks_in_flower():
    """
    Let Flower's NumPyClient send and receive CKKS ciphertexts, in every process that imports `utils`.

    The original QFL notebooks patched these converters in the notebook process only. Flower's Ray workers
    import flwr afresh, so their clients crashed on np.save(CKKSTensor) as soon as updates were really encrypted.
    """
    import flwr.common.parameter as flwr_parameter

    flwr_parameter.ndarray_to_bytes = ndarray_to_bytes_ckks
    flwr_parameter.bytes_to_ndarray = bytes_to_ndarray_ckks


def aggregate_custom(results: List[Tuple[NDArrays, int]], log_path: Optional[str] = None) -> NDArrays:
    """
    FedAvg: weighted average of client weights, valid for numpy arrays and CKKS ciphertexts.

    Each client's weights are scaled once by n_k / n (one CKKS level) and then summed,
    so the server never needs to decrypt.
    """
    start_time = time.time()
    num_examples_total = sum(num_examples for _, num_examples in results)
    weighted_weights = [
        [layer * (num_examples / num_examples_total) for layer in weights] for weights, num_examples in results
    ]
    weights_prime = [reduce(operator.add, layer_updates) for layer_updates in zip(*weighted_weights)]
    log_metric(f"[S Agg] Time: {time.time() - start_time:.4f}s", log_path)
    return weights_prime
