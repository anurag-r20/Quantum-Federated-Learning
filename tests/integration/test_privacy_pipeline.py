"""Integration tests: real Flower/Ray simulations on a synthetic MRI set, then the attacks on what the server
recorded. Marked `integration` (about 30 s each); run with  pytest -m integration.

Oracles: the private image the client used (saved for scoring only) and TenSEAL's refusal to decrypt without
the secret key.
"""

from __future__ import annotations

import pytest
import torch

import gia_attack as gia
from federated import FLConfig, ensure_keys, load_context, run_federated
from models import MRINet

pytestmark = pytest.mark.integration


def pipeline_config(tmp_path, data_path, he: bool) -> FLConfig:
    return FLConfig(he=he, data_path=str(data_path), img_size=16, num_clients=3, rounds=2, local_epochs=1,
                    capture_round=2, capture_batch_size=1, out_dir=str(tmp_path / "out"),
                    keys_dir=str(tmp_path / "keys"), cpus_per_client=1)


def test__plaintext_pipeline__given_a_fedsgd_capture_round__lets_the_server_reconstruct_a_private_image(
    tmp_path, make_mri_cache
) -> None:
    # Arrange
    cfg = pipeline_config(tmp_path, make_mri_cache(tmp_path), he=False)

    # Act
    summary = run_federated(cfg)

    # Assert
    assert [r["round"] for r in summary["rounds"]] == [1, 2]
    observation = gia.load_observation(tmp_path / "out" / "server_observation.pkl")
    attack = gia.attack_plaintext(observation, MRINet(num_classes=4, img_size=16), "0",
                                  gia.GIAConfig(iterations=600, restarts=1), data_shape=(1, 16, 16))
    truth = torch.load(tmp_path / "out" / "gia_ground_truth" / "client0.pt")
    assert attack["labels"].tolist() == truth["labels"].tolist()
    assert gia.evaluate_reconstruction(attack["images"], truth["images"])["mean_psnr"] > 25


def test__fhe_pipeline__given_encrypted_updates__blocks_the_server_and_still_trains(tmp_path, make_mri_cache) -> None:
    # Arrange
    cfg = pipeline_config(tmp_path, make_mri_cache(tmp_path), he=True)

    # Act
    summary = run_federated(cfg)

    # Assert
    observation = gia.load_observation(tmp_path / "out" / "server_observation.pkl")
    assert sorted(observation["updates"]) == ["0", "1", "2"]
    report = gia.attempt_on_encrypted(observation, load_context(ensure_keys(cfg.keys_dir)[1]), "0")
    assert report["status"] == "blocked"
    assert summary["rounds"][0]["upload_mb_per_client"] > 1  # ciphertexts, not 0.03 MB of plaintext weights
    assert (tmp_path / "out" / "final_model.pt").exists()
