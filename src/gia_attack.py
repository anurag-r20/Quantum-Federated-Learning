"""
Gradient inversion attack (GIA) run by an honest-but-curious federated-learning server.

The server knows the global weights it sent, the client's returned weights and the round's
hyperparameters (it chose them). In a FedSGD round the update is exactly -lr * gradient, so
the server recovers the client's gradient and searches for images that reproduce it:

  1. label: recovered analytically from the last layer's bias gradient (iDLG, Zhao et al. 2020)
  2. features: for batch size 1 the input of the first fully-connected layer is recovered
     exactly as grad_W / grad_b (Geiping et al. 2020, Prop. 1; Phong et al. 2017)
  3. image: Adam on the pixels minimising 1 - cos(grad(x), observed grad) + feature error
     + total variation, with signed gradients, restarts and box constraints (Geiping et al. 2020)

When updates are CKKS-encrypted the server holds only the public context, so step 0 (reading
the update) already fails: TenSEAL refuses to decrypt without the secret key.
"""
import json
import os
import pickle
import time
from dataclasses import asdict, dataclass
from io import BytesIO

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tenseal as ts
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from utils import data_setup, security


@dataclass
class GIAConfig:
    iterations: int = 3000
    lr: float = 0.1
    restarts: int = 2
    tv_weight: float = 5e-3
    feature_weight: float = 1.0       # analytic first-FC-layer input matching (batch size 1 only)
    signed_gradients: bool = True
    seed: int = 0


# ------------------------------------------------------------------ observation -> gradient

def load_observation(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def observed_gradient(observation, cid, model):
    """Server-side: gradient of a FedSGD update, g = (w_global - w_client) / lr. Plaintext updates only."""
    update = observation["updates"][str(cid)]
    if update["tensor_type"] == security.TENSOR_TYPE_CKKS:
        raise PermissionError("update is CKKS-encrypted; use attempt_on_encrypted()")
    lr = float(observation["fit_config"]["lr"])
    load = lambda b: np.load(BytesIO(b), allow_pickle=False).astype(np.float64)
    names = list(model.state_dict().keys())
    grads = {name: torch.tensor((load(g) - load(c)) / lr, dtype=torch.float32)
             for name, g, c in zip(names, observation["global_tensors"], update["tensors"])}
    global_state = {name: torch.tensor(load(g), dtype=torch.float32)
                    for name, g in zip(names, observation["global_tensors"])}
    return grads, global_state


def _linear_layers(model):
    return [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def recover_labels_idlg(grads, model):
    """iDLG: with softmax cross-entropy the true class is the only negative entry of the last bias gradient."""
    name, _ = _linear_layers(model)[-1]
    return torch.argmin(grads[f"{name}.bias"]).view(1)


def recover_fc_input(grads, model):
    """Least-squares solution of grad_W = grad_b * h^T: exact input of the first FC layer for batch size 1."""
    name, _ = _linear_layers(model)[0]
    g_w, g_b = grads[f"{name}.weight"], grads[f"{name}.bias"]
    return (g_b @ g_w) / (g_b @ g_b)


# ------------------------------------------------------------------ reconstruction

def total_variation(x):
    return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()


def cosine_gradient_loss(dummy_grads, target_grads):
    dot = sum((d * t).sum() for d, t in zip(dummy_grads, target_grads))
    dummy_norm = sum(d.pow(2).sum() for d in dummy_grads).sqrt()
    target_norm = sum(t.pow(2).sum() for t in target_grads).sqrt()
    return 1 - dot / (dummy_norm * target_norm + 1e-12)


def reconstruct(model, grads, labels, data_shape, cfg: GIAConfig, fc_input=None, log_every=500):
    """Optimise dummy images (in model input space [-1, 1]) so their gradient matches the observed one."""
    model.eval()
    names = [n for n, _ in model.named_parameters()]
    params = [p for _, p in model.named_parameters()]
    target = [grads[n] for n in names]
    batch = len(labels)

    first_fc = _linear_layers(model)[0][1]
    captured = {}
    hook = first_fc.register_forward_pre_hook(lambda module, inputs: captured.__setitem__("h", inputs[0]))
    milestones = [int(cfg.iterations * f) for f in (3 / 8, 5 / 8, 7 / 8)]
    generator = torch.Generator().manual_seed(cfg.seed)
    best = {"loss": float("inf")}
    try:
        for restart in range(cfg.restarts):
            x = (0.5 * torch.randn((batch,) + tuple(data_shape), generator=generator)).clamp(-1, 1)
            x.requires_grad_(True)
            optimizer = torch.optim.Adam([x], lr=cfg.lr)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
            history = []
            for it in range(cfg.iterations):
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x), labels)
                dummy_grads = torch.autograd.grad(loss, params, create_graph=True)
                match = cosine_gradient_loss(dummy_grads, target)
                if fc_input is not None and cfg.feature_weight > 0:
                    h = captured["h"].reshape(batch, -1).mean(0)
                    match = match + cfg.feature_weight * (h - fc_input).pow(2).sum() / fc_input.pow(2).sum()
                objective = match + cfg.tv_weight * total_variation(x)
                x.grad, = torch.autograd.grad(objective, x)
                if cfg.signed_gradients:
                    x.grad.sign_()
                optimizer.step()
                scheduler.step()
                with torch.no_grad():
                    x.clamp_(-1, 1)
                if it % 50 == 0 or it == cfg.iterations - 1:
                    history.append(float(match))
                if log_every and (it % log_every == 0 or it == cfg.iterations - 1):
                    print(f"    restart {restart + 1}/{cfg.restarts} it {it:5d} | gradient-match loss {float(match):.5f}")
            if history[-1] < best["loss"]:
                best = {"loss": history[-1], "images": x.detach().clone(), "history": history, "restart": restart}
    finally:
        hook.remove()
    return best


# ------------------------------------------------------------------ evaluation (uses ground truth)

def _gaussian_window(size=11, sigma=1.5):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(a, b):
    """Mean SSIM per image for (N, 1, H, W) tensors in [0, 1]."""
    window = _gaussian_window()
    mu_a, mu_b = F.conv2d(a, window), F.conv2d(b, window)
    var_a = F.conv2d(a * a, window) - mu_a ** 2
    var_b = F.conv2d(b * b, window) - mu_b ** 2
    cov = F.conv2d(a * b, window) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return ssim_map.mean(dim=(1, 2, 3))


def psnr(a, b):
    mse = ((a - b) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
    return 10 * torch.log10(1 / mse)


def evaluate_reconstruction(reconstruction, ground_truth):
    """PSNR/SSIM in pixel space; for batches, each reconstruction is matched to its best ground-truth image."""
    rec, gt = data_setup.to_pixels(reconstruction), data_setup.to_pixels(ground_truth)
    cost = torch.stack([-psnr(rec[i:i + 1].expand_as(gt), gt) for i in range(len(rec))]).numpy()
    rec_idx, gt_idx = linear_sum_assignment(cost)
    rec, gt = rec[rec_idx], gt[gt_idx]
    return {"psnr": psnr(rec, gt).tolist(), "ssim": ssim(rec, gt).tolist(),
            "mean_psnr": float(psnr(rec, gt).mean()), "mean_ssim": float(ssim(rec, gt).mean()),
            "order": gt_idx.tolist()}


# ------------------------------------------------------------------ attacks on a recorded round

def attack_plaintext(observation, model, cid="0", cfg: GIAConfig = GIAConfig(), known_labels=None,
                     data_shape=(1, 64, 64)):
    """Full server-side attack on one client's plaintext FedSGD update."""
    grads, global_state = observed_gradient(observation, cid, model)
    model.load_state_dict(global_state)
    batch = int(observation["updates"][str(cid)]["num_examples"])
    if batch == 1:
        labels, label_source = recover_labels_idlg(grads, model), "iDLG (analytic)"
        fc_input = recover_fc_input(grads, model)
    else:
        if known_labels is None:
            raise ValueError("batch size > 1: pass known_labels (label recovery is only analytic for batch size 1)")
        labels, label_source, fc_input = torch.as_tensor(known_labels), "assumed known (batch > 1)", None
    start = time.time()
    best = reconstruct(model, grads, labels, data_shape, cfg, fc_input=fc_input)
    return {"images": best["images"], "labels": labels, "label_source": label_source,
            "gradient_match_loss": best["loss"], "history": best["history"],
            "seconds": round(time.time() - start, 1), "batch_size": batch, "client": str(cid)}


def attempt_on_encrypted(observation, public_ctx, cid="0"):
    """What the server can do with a CKKS-encrypted update: nothing readable without the secret key."""
    update = observation["updates"][str(cid)]
    report = {"client": str(cid), "tensor_type": update["tensor_type"],
              "ciphertext_mb": round(sum(len(t) for t in update["tensors"]) / 2 ** 20, 2)}
    try:
        encrypted = ts.ckks_tensor_from(public_ctx, update["tensors"][0])
        encrypted.decrypt()
        report.update(status="NOT BLOCKED", error=None)
    except Exception as error:  # TenSEAL raises ValueError: no secret key in the server's context
        report.update(status="blocked", error=f"{type(error).__name__}: {error}")
    return report


def ciphertext_preview(observation, cid="0", side=64):
    """First side*side bytes of the client's first ciphertext, shown as an image (what the server 'sees')."""
    raw = np.frombuffer(observation["updates"][str(cid)]["tensors"][0][:side * side], dtype=np.uint8)
    return raw.reshape(side, side)


# ------------------------------------------------------------------ plotting

def plot_reconstruction(ground_truth, result, metrics, classes, path=None, title=None):
    """Private images (top / left) next to their matched reconstructions."""
    gt = data_setup.to_pixels(ground_truth)[metrics["order"]]   # metrics[i] pairs reconstruction i with gt order[i]
    rec = data_setup.to_pixels(result["images"])
    n = len(gt)
    if n == 1:
        fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.3), squeeze=False)
        cells = [(axes[0, 0], axes[0, 1])]
    else:
        fig, axes = plt.subplots(2, n, figsize=(2.4 * n + 0.6, 5.2), squeeze=False)
        cells = [(axes[0, i], axes[1, i]) for i in range(n)]
    for i, (ax_gt, ax_rec) in enumerate(cells):
        ax_gt.imshow(gt[i, 0], cmap="gray", vmin=0, vmax=1)
        ax_gt.set_title("private image" if n == 1 else f"private #{metrics['order'][i]}", fontsize=10)
        ax_rec.imshow(rec[i, 0], cmap="gray", vmin=0, vmax=1)
        ax_rec.set_title(f"reconstruction\nPSNR {metrics['psnr'][i]:.1f} dB, SSIM {metrics['ssim'][i]:.2f}", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(title or f"Gradient inversion (batch {n}, label: {result['label_source']}"
                 f" -> {[classes[int(l)] for l in result['labels']]})", fontsize=11)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def save_report(report, path):
    serializable = {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in report.items() if k != "images"}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
