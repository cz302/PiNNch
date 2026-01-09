# -*- coding: utf-8 -*-
"""
Torch regression test: notebook-params PiNet2 must beat the zero baseline on rMD17 aspirin.

External requirement:
- aspirin-1000.npz available locally (path via env ASPIRIN_NPZ or repo root)

Everything else is self-contained and follows the TF reference test structure:
- deterministic split: first 800 train, last 200 eval
- builds YAML datasets via pinn.io.write_tfrecord (Torch Colab style)
- trains via pinn.train_and_evaluate(train_yml=..., eval_yml=...)
- evaluates MAE in the same label space used by the torch runtime metrics:
    E_err = (E_pred/e_unit) - E_true
    F_err = (F_pred/e_unit) - F_true
  and compares MAE vs the zero predictor baseline (E_pred=0, F_pred=0).
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pytest

import torch
import pinn
from pinn.io import load_numpy, write_tfrecord
from pinn.torch.input_pipeline import TorchDataOptions, make_torch_dataloader_from_yml
from pinn.torch.runtime import move_pinn_batch_to_device, maybe_compile_model


def _find_aspirin_npz() -> Path:
    """Locate aspirin-1000.npz via env or by searching upward from tests/."""
    env = os.environ.get("ASPIRIN_NPZ", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p

    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        cand = p / "aspirin-1000.npz"
        if cand.is_file():
            return cand

    pytest.skip("aspirin-1000.npz not found. Set ASPIRIN_NPZ=/path/to/aspirin-1000.npz")


def _zero_baseline_mae(eval_raw: list[dict]) -> tuple[float, float]:
    """
    Zero predictor baseline (E_pred=0, F_pred=0), in dataset label units.

    Energy baseline is computed *per atom* to match per-atom energy metrics.
    Force baseline is per component (Fx/Fy/Fz), unchanged.

    Returns:
      energy_mae_per_atom, force_mae_per_component
    """
    e_per_atom = []
    f_all = []

    for ex in eval_raw:
        e = float(np.asarray(ex["e_data"]).reshape(-1)[0])  # scalar energy label
        f = np.asarray(ex["f_data"], dtype=float)          # (N,3)

        n = int(f.shape[0])
        if n <= 0:
            continue

        e_per_atom.append(abs(e) / n)
        f_all.append(f)

    if not e_per_atom:
        raise ValueError("No valid frames for baseline (empty or zero-atom entries).")

    f_all = np.concatenate(f_all, axis=0)  # (sum_N, 3)
    return float(np.mean(e_per_atom)), float(np.mean(np.abs(f_all)))


def _get_nl_builder_from_model(model):
    """
    Minimal, robust NL-builder getter (matches what we used in runtime.py fixes).

    Expected layout for torch PiNet/PiNet2:
      model.network.preprocess.nl  (or similar wrappers)
    """
    candidates = [
        getattr(model, "network", None),
        getattr(model, "net", None),
        getattr(model, "model", None),
        model,
    ]
    for obj in candidates:
        if obj is None:
            continue
        for pre_name in ("preprocess", "preprocess_layer", "pre", "pp"):
            pre = getattr(obj, pre_name, None)
            if pre is not None and hasattr(pre, "nl"):
                return pre.nl
        if hasattr(obj, "nl"):
            return obj.nl
    raise AttributeError(
        "Could not find neighbor-list builder on model. "
        "Expected something like model.network.preprocess.nl"
    )


def _torch_eval_mae_from_yml(
    *,
    model,
    params: dict,
    eval_yml: str,
    scratch_dir: str,
    device: str,
    eval_steps: int,
    batch_size_eval: int,
) -> tuple[float, float]:
    """Compute (energy_mae, force_mae_per_component) on eval set (Torch YAML pipeline)."""
    mp = params.get("model", {}).get("params", {}) or {}
    e_unit = float(mp.get("e_unit", 1.0))
    use_force = bool(mp.get("use_force", True))

    netp = params["network"]["params"]
    atom_types = list(netp["atom_types"])
    rc = float(netp["rc"])

    opts = TorchDataOptions(
        batch_size=int(batch_size_eval),
        shuffle_buffer=0,
        atom_types=atom_types,
        rc=rc,
        scratch_dir=scratch_dir,
        cache=True,
        cache_ram=True,
        device="cpu",   # CPU pipeline; move to GPU in this function
        preprocess=True,
    )

    nl_builder = _get_nl_builder_from_model(model)
    eval_loader = make_torch_dataloader_from_yml(
        eval_yml, dataset_role="eval", opts=opts, nl_builder=nl_builder
    )
    eval_it = iter(eval_loader)

    dev = torch.device(device)
    if hasattr(model, "to"):
        model.to(dev)
    if hasattr(model, "eval"):
        model.eval()

    E_abs_sum = 0.0
    E_count = 0
    F_abs_sum = 0.0
    F_count = 0

    for _ in range(int(eval_steps)):
        try:
            batch = next(eval_it)
        except StopIteration:
            break

        if "e_data" not in batch or "f_data" not in batch:
            raise KeyError("Torch YAML batch must include 'e_data' and 'f_data'.")

        # Labels to device
        E_true = batch["e_data"].to(dev, non_blocking=True)
        F_true = batch["f_data"].to(dev, non_blocking=True)

        # Inputs to device + coord requires_grad on device (canonical)
        tensors = {k: v for k, v in batch.items() if k not in ("e_data", "f_data")}
        tensors = move_pinn_batch_to_device(tensors, dev)
        coord = tensors["coord"]

        # Forward
        E_pred = model(tensors)

        # Accept (B,1) or (B,)
        if E_pred.ndim == 2 and E_pred.shape[1] == 1:
            E_pred = E_pred[:, 0]
        elif E_pred.ndim != 1:
            raise ValueError(f"Expected E_pred shape (B,) or (B,1), got {tuple(E_pred.shape)}")

        # Number of structures
        B = int(E_true.shape[0])  # ok for your current dataset
        # Safer alternative (uncomment if you want):
        # B = int(tensors["ind_1"][:, 0].max().item()) + 1

        counts = (
            torch.bincount(tensors["ind_1"][:, 0], minlength=B)
            .to(E_true.dtype)
            .clamp_min(1.0)
        )

        # Per-atom energy MAE in label space
        E_abs_sum += torch.abs((E_pred / counts) / e_unit - (E_true / counts)).sum().item()
        E_count += int(E_true.numel())

        if use_force:
            dE_dR = torch.autograd.grad(E_pred.sum(), coord, create_graph=False, retain_graph=False)[0]
            F_pred = -dE_dR

            if F_true.ndim == 3:
                F_true = F_true.reshape(-1, 3)
            if F_pred.ndim == 3:
                F_pred = F_pred.reshape(-1, 3)

            F_abs_sum += torch.abs((F_pred / e_unit) - F_true).sum().item()
            F_count += int(F_true.numel())

    e_mae = float(E_abs_sum / max(E_count, 1))
    f_mae = float(F_abs_sum / max(F_count, 1)) if use_force else 0.0
    return e_mae, f_mae


def test_aspirin_rmd17_torch_notebook_params_beats_zero_baseline(tmp_path, monkeypatch):
    monkeypatch.setenv("PINN_BACKEND", "torch")

    npz_path = _find_aspirin_npz()

    # ---- Load and split deterministically: first 800 train, last 200 eval ----
    ds_all = load_numpy(np.load(npz_path))
    raw = list(ds_all.as_numpy_iterator())
    assert len(raw) >= 1000, f"Expected >=1000 frames, got {len(raw)}"

    train_raw = raw[:800]
    eval_raw = raw[800:1000]

    # ---- Params: match your TF reference test (PiNet2 + rank=3) ----
    # (only add model_dir for pytest isolation)
    params = {
        "model": {
            "name": "potential_model",
            "params": {
                "e_loss_multiplier": 1.0,
                "f_loss_multiplier": 10.0,
                "log_e_per_atom": True,
                "use_e_per_atom": True,
                "use_force": True,
            },
        },
        "network": {
            "name": "PiNet2",
            "params": {
                "atom_types": [1, 6, 7, 8],
                "basis_type": "gaussian",
                "depth": 5,
                "n_basis": 10,
                "pi_nodes": [16],
                "ii_nodes": [16, 16],
                "pp_nodes": [16, 16],
                "out_nodes": [16],
                "rank": 3,
                "rc": 4.5,
                "torsion_boost": False,
            },
        },
        "optimizer": {
            "class_name": "Adam",
            "config": {
                "global_clipnorm": 0.01,
                "learning_rate": {
                    "class_name": "ExponentialDecay",
                    "config": {
                        "decay_rate": 0.994,
                        "decay_steps": 100000,
                        "initial_learning_rate": 5.0e-05,
                    },
                },
            },
        },
        "model_dir": str(tmp_path / "pinet2_aspirin_torch_notebook_params"),
    }

    # ---- Baseline (label units) ----
    e0, f0 = _zero_baseline_mae(eval_raw)

    # ---- Create YAML datasets (Torch Colab style: pinn.io.write_tfrecord) ----
    # We avoid the Colab bug (train.take(800), eval.take(200)) by explicitly slicing.
    train_dict = {
        "coord": np.asarray([r["coord"] for r in train_raw], dtype=np.float32),
        "elems": np.asarray([r["elems"] for r in train_raw], dtype=np.int32),
        "e_data": np.asarray([r["e_data"] for r in train_raw], dtype=np.float32),
        "f_data": np.asarray([r["f_data"] for r in train_raw], dtype=np.float32),
    }
    eval_dict = {
        "coord": np.asarray([r["coord"] for r in eval_raw], dtype=np.float32),
        "elems": np.asarray([r["elems"] for r in eval_raw], dtype=np.int32),
        "e_data": np.asarray([r["e_data"] for r in eval_raw], dtype=np.float32),
        "f_data": np.asarray([r["f_data"] for r in eval_raw], dtype=np.float32),
    }

    train_yml = str(tmp_path / "aspirin-train.yml")
    eval_yml = str(tmp_path / "aspirin-eval.yml")

    write_tfrecord(train_yml, load_numpy(train_dict))
    write_tfrecord(eval_yml, load_numpy(eval_dict))

    # ---- Train via torch runtime (YAML pipeline) ----
    model = pinn.get_model(params)
    model = maybe_compile_model(model)

    num_train_steps = int(os.environ.get("ASPIRIN_TORCH_MAX_STEPS", "2000"))
    eval_steps = int(os.environ.get("ASPIRIN_TORCH_EVAL_STEPS", "200"))
    batch_size_train = int(os.environ.get("ASPIRIN_TORCH_BATCH_TRAIN", "2"))
    batch_size_eval = int(os.environ.get("ASPIRIN_TORCH_BATCH_EVAL", "2"))
    shuffle_buffer = int(os.environ.get("ASPIRIN_TORCH_SHUFFLE_BUFFER", "1000"))

    scratch_dir = str(tmp_path / "torch_cache")
    Path(scratch_dir).mkdir(parents=True, exist_ok=True)

    metrics = pinn.train_and_evaluate(
        model=model,
        params=params,
        data=None,
        train_yml=train_yml,
        eval_yml=eval_yml,
        max_steps=int(num_train_steps),
        eval_steps=int(eval_steps),
        batch_size_train=int(batch_size_train),
        batch_size_eval=int(batch_size_eval),
        shuffle_buffer=int(shuffle_buffer),
        preprocess=True,
        cache=True,
        cache_ram=True,
        scratch_dir=scratch_dir,
    )
    # useful for debugging with `pytest -s`
    print("torch train_and_evaluate metrics:", metrics)

    # ---- Evaluate MAE on eval set (same label space as runtime metrics) ----
    device = "cuda" if torch.cuda.is_available() else "cpu"

    e_mae, f_mae = _torch_eval_mae_from_yml(
        model=model,
        params=params,
        eval_yml=eval_yml,
        scratch_dir=scratch_dir,
        device=device,
        eval_steps=eval_steps,
        batch_size_eval=batch_size_eval,
    )

    # Convert kcal/mol -> meV (per molecule) using 1 kcal/mol = 43.3641153088 meV
    KCALMOL_TO_MEV = 43.3641153088
    print(f"Energy MAE: {e_mae * KCALMOL_TO_MEV:.2f} meV")
    print(f"Force  MAE: {f_mae * KCALMOL_TO_MEV:.2f} meV/Å")

    # ---- Assert: beats baseline by 10% ----
    assert e_mae < 0.90 * e0, f"Energy MAE {e_mae} did not beat zero baseline {e0} by 10%"
    assert f_mae < 0.90 * f0, f"Force MAE {f_mae} did not beat zero baseline {f0} by 10%"
