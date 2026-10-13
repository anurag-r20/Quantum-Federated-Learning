"""
Model inversion attack (MIA) on the final federated model.

FHE protects updates while training, but every client receives (and decrypts) the final global
model. Anyone holding it can search for inputs the model assigns to a class with high
confidence and so recover what the private training data of that class looks like.

Following generative-prior model inversion (Zhang et al., CVPR 2020) in a lightweight form, the
attacker searches a PCA "brain image space" fitted on public, unlabelled scans (here the
held-out Testing split) instead of raw pixels, which keeps reconstructions MRI-like:

    x(z) = mean + (z * std) @ components,   minimise  CE(model(x(z)), class) + prior_weight * |z|^2

Success is scored by an independent evaluation classifier trained on the private data
(attack accuracy, as in Zhang et al.) and by the closest private training image.
"""
import json
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from models import EvaluationNet
from utils import data_setup, engine
from gia_attack import ssim


@dataclass
class MIAConfig:
    n_components: int = 100
    iterations: int = 1500
    lr: float = 0.05
    prior_weight: float = 0.05
    restarts: int = 8
    seed: int = 0


# ------------------------------------------------------------------ prior and evaluation model

def fit_pca_prior(public_images, n_components=100):
    """PCA of public (unlabelled) images in model input space."""
    flat = public_images.reshape(len(public_images), -1).double()
    mean = flat.mean(0)
    _, singular_values, components = torch.linalg.svd(flat - mean, full_matrices=False)
    std = singular_values[:n_components] / (len(flat) - 1) ** 0.5
    return {"mean": mean.float(), "components": components[:n_components].float(), "std": std.float(),
            "shape": tuple(public_images.shape[1:])}


def decode(prior, z):
    x = prior["mean"] + (z * prior["std"]) @ prior["components"]
    return x.reshape((len(z),) + prior["shape"]).clamp(-1, 1)


def train_evaluation_classifier(x_train, y_train, x_test, y_test, epochs=8, seed=0):
    """Independent classifier (different architecture) used only to score MIA reconstructions."""
    torch.manual_seed(seed)
    model = EvaluationNet(num_classes=4)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        loss, acc = engine.train_step(model, loader, torch.nn.CrossEntropyLoss(), optimizer, "cpu")
        print(f"    evaluation classifier epoch {epoch + 1}/{epochs}: train acc {acc:.1f}%")
    model.eval()
    _, test_acc, *_ = engine.test(model, DataLoader(TensorDataset(x_test, y_test), batch_size=256),
                                  torch.nn.CrossEntropyLoss(), "cpu")
    return model, float(test_acc)


# ------------------------------------------------------------------ attack

def invert_class(model, target_class, prior, cfg: MIAConfig):
    """Search the prior's latent space for an image the target model assigns to target_class."""
    model.eval()
    generator = torch.Generator().manual_seed(cfg.seed + target_class)
    z = (0.5 * torch.randn(cfg.restarts, len(prior["std"]), generator=generator)).requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=cfg.lr)
    labels = torch.full((cfg.restarts,), target_class)
    for _ in range(cfg.iterations):
        optimizer.zero_grad()
        ce = F.cross_entropy(model(decode(prior, z)), labels, reduction="none")
        loss = ce + cfg.prior_weight * z.pow(2).mean(1)
        loss.sum().backward()
        optimizer.step()
    with torch.no_grad():
        images = decode(prior, z)
        confidence = torch.softmax(model(images), 1)[:, target_class]
        objective = -torch.log(confidence + 1e-12) + cfg.prior_weight * z.pow(2).mean(1)
        best = int(torch.argmin(objective))
    return images[best:best + 1], float(confidence[best])


def nearest_private_images(image, private_images, private_labels, k=10):
    """Most similar private training images (SSIM in pixel space)."""
    pixels = data_setup.to_pixels(private_images)
    scores = torch.cat([ssim(data_setup.to_pixels(image).expand(len(chunk), -1, -1, -1), chunk)
                        for chunk in pixels.split(1024)])
    top = torch.topk(scores, k)
    return top.indices, top.values, private_labels[top.indices]


def run_mia(model, prior, evaluation_model, private_images, private_labels, classes, cfg: MIAConfig = MIAConfig()):
    """Invert every class of `model`; score each reconstruction with the evaluation classifier."""
    start = time.time()
    results = []
    for c, name in enumerate(classes):
        image, confidence = invert_class(model, c, prior, cfg)
        with torch.no_grad():
            eval_probs = torch.softmax(evaluation_model(image), 1)[0]
        nn_idx, nn_ssim, nn_labels = nearest_private_images(image, private_images, private_labels)
        results.append({
            "class": name,
            "image": image,
            "target_model_confidence": round(confidence, 4),
            "evaluation_classifier_prediction": classes[int(eval_probs.argmax())],
            "evaluation_classifier_confidence": round(float(eval_probs[c]), 4),
            "nearest_private_index": int(nn_idx[0]),
            "nearest_private_class": classes[int(nn_labels[0])],
            "nearest_private_ssim": round(float(nn_ssim[0]), 3),
            "top10_neighbours_in_class": float((nn_labels == c).float().mean()),
        })
        print(f"    {name:10s} target-model conf {confidence:.3f} | evaluation classifier says "
              f"{results[-1]['evaluation_classifier_prediction']} ({float(eval_probs[c]):.2f}) | "
              f"closest private scan: {results[-1]['nearest_private_class']} (SSIM {float(nn_ssim[0]):.2f})")
    attack_accuracy = float(np.mean([r["evaluation_classifier_prediction"] == r["class"] for r in results]))
    return {"per_class": results, "attack_accuracy": attack_accuracy, "seconds": round(time.time() - start, 1)}


# ------------------------------------------------------------------ reporting

def plot_mia(reports, private_images, classes, path=None):
    """reports: {"plaintext FL": report, "FHE FL": report}. One row per model plus the closest private scans."""
    rows = list(reports.items())
    fig, axes = plt.subplots(len(rows) + 1, len(classes), figsize=(2.6 * len(classes), 3.1 * (len(rows) + 1)))
    for r, (label, report) in enumerate(rows):
        for c, result in enumerate(report["per_class"]):
            ax = axes[r, c]
            ax.imshow(data_setup.to_pixels(result["image"])[0, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{label}: {result['class']}\nmodel conf {result['target_model_confidence']:.2f}, "
                         f"eval: {result['evaluation_classifier_prediction']}", fontsize=8)
    first = rows[0][1]["per_class"]
    for c, result in enumerate(first):
        ax = axes[-1, c]
        ax.imshow(data_setup.to_pixels(private_images[result["nearest_private_index"]])[0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"closest private scan\n({result['nearest_private_class']}, SSIM {result['nearest_private_ssim']:.2f})",
                     fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Model inversion of the final global model (class-representative reconstructions)", fontsize=11)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def save_report(report, path):
    serializable = {**report, "per_class": [{k: v for k, v in r.items() if k != "image"} for r in report["per_class"]]}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
