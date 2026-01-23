# -*- coding: utf-8 -*-
"""
Torch regression test: PiNet2 on liquid water (RuNNer input.data) must beat a zero baseline.

Key difference vs aspirin_md17_torch:
- Per-atom energy is not meaningful for liquid water.
- We report energy as *per-water* energy, assuming N_WATER water molecules per frame
  (default N_WATER=64, override via env WATER_TORCH_N_WATER).

External requirement:
- Local RuNNer-format dataset file "input.data" (path via env WATER_INPUT_DATA or repo root).

Test outline:
- Load RuNNer dataset via pinn.io.load_runner (coords in Å, energies in Hartree, forces in Hartree/Å).
- Build small train/eval subsets and write them as TFRecord+YAML via pinn.io.write_tfrecord.
- Train via pinn.train_and_evaluate(..., backend=torch).
- Evaluate MAE in the same label space as the torch runtime metrics:
    E_err = (E_pred/e_unit) - E_true
    F_err = (F_pred/e_unit) - F_true
  but with E_true and E_pred converted to per-water values: divide by N_WATER.
- Assert it beats the zero predictor baseline by a configurable fraction.

Environment knobs (optional):
- WATER_INPUT_DATA: path to input.data
- WATER_TORCH_MAX_STEPS: training steps (default 2000)
- WATER_TORCH_EVAL_STEPS: evaluation steps/batches (default 200)
- WATER_TORCH_BATCH_TRAIN: train batch size (default 1)
- WATER_TORCH_BATCH_EVAL: eval batch size (default 1))
- WATER_TORCH_SHUFFLE_SEED: split seed (default 1)
- WATER_TORCH_BEAT_BASELINE_FRAC: required relative improvement vs zero baseline (default 0.03 = 2%)
- WATER_TORCH_N_WATER: number of water molecules per frame (default 64)
"""

from __future__ import annotations
from pinn.torch.prof import walltime

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch

import pinn
from pinn.io import load_runner, write_tfrecord
from pinn.utils import init_params
from pathlib import Path


def debug_one_batch_pbc_from_yml(*, yml_path: Path, params: dict, scratch_dir: Path) -> None:
    """
    Print PBC geometry diagnostics for the first batch produced from a TFRecord YAML.

    Uses ONLY PiNNch APIs:
      - pinn.io.build_dataset.build_dataset + BuildOptions (backend="torch")
      - pinn.torch.input_pipeline.TorchDataOptions + _iter_batches_from_examples
      - pinn.torch.runtime._get_nl_builder_from_model

    Checks:
      - MIC dist stats on all edges (fractional rounding)
      - edges with dist > rc
      - coord span vs cell length (unwrapping indicator)
      - |F_true| magnitude stats
      - optional |F_pred| magnitude stats via autograd

    Env:
      - WATER_TORCH_PBC_DEBUG_PRED=1 enables predicted-force stats
    """
    import os
    import numpy as np
    import torch

    import pinn
    from pinn.io.build_dataset import build_dataset, BuildOptions
    from pinn.torch.input_pipeline import TorchDataOptions, _iter_batches_from_examples
    import pinn.torch.runtime as rt

    rc = float(params["network"]["params"]["rc"])
    atom_types = list(params["network"]["params"]["atom_types"])
    hartree_to_ev = globals().get("HARTREE_TO_EV", 27.2114)

    # Build a real model and locate its NL builder exactly as runtime does
    model = pinn.get_model(params)
    nl_builder = rt._get_nl_builder_from_model(model)

    # Build dataset examples exactly like the torch input path
    built_examples = build_dataset(
        str(yml_path),
        options=BuildOptions(
            backend="torch",
            cache=False,  # avoid cached artifacts masking code changes
            scratch_dir=str(scratch_dir),
            cache_ram=False,
        ),
        dataset_role="debug",
    )

    opts = TorchDataOptions(
        batch_size=1,
        shuffle_buffer=0,
        atom_types=atom_types,
        rc=rc,
        scratch_dir=str(scratch_dir),
        cache=False,
        cache_ram=False,
        device="cpu",
        preprocess=True,
    )

    it = _iter_batches_from_examples(built_examples, opts=opts, nl_builder=nl_builder)
    batch = next(it)

    # ------------------- import path diagnostics -------------------
    print("\n[PBC_DEBUG] import paths:")
    import pinn.io.torch.preprocess_fns as pf
    import pinn.networks.pinet_torch as pt
    print("  preprocess_fns:", pf.__file__)
    print("  pinet_torch   :", pt.__file__)
    print("  runtime       :", rt.__file__)

    # ------------------- basic batch diagnostics -------------------
    print("\n[PBC_DEBUG] batch keys:", sorted(batch.keys()))
    coord = batch["coord"]
    ind_2 = batch["ind_2"]
    cell = batch.get("cell", None)
    shift = batch.get("shift", None)

    print("[PBC_DEBUG] coord:", tuple(coord.shape), coord.dtype, coord.device)
    print("[PBC_DEBUG] ind_2:", tuple(ind_2.shape), ind_2.dtype, ind_2.device)

    if cell is None:
        print("[PBC_DEBUG] cell=None (non-PBC) -> nothing to check")
        return

    print("[PBC_DEBUG] cell:", tuple(cell.shape), cell.dtype, cell.device)
    if shift is None:
        print("[PBC_DEBUG] shift: None")
    else:
        sh = shift.detach().cpu().numpy()
        print("[PBC_DEBUG] shift:", tuple(shift.shape), shift.dtype,
              "min", sh.min(axis=0), "max", sh.max(axis=0))

    # ------------------- |F_true| magnitude stats -------------------
    if "f_data" in batch:
        f_true = batch["f_data"]
        if torch.is_tensor(f_true) and f_true.numel() > 0:
            f_norm = torch.linalg.norm(f_true, dim=-1)  # (N,)
            fn = f_norm.detach().cpu()
            p50 = float(fn.median())
            p90 = float(torch.quantile(fn, 0.9))
            print(
                "[PBC_DEBUG] |F_true| (Hartree/Å) stats: "
                f"min={float(fn.min()):.6g}  p50={p50:.6g}  p90={p90:.6g}  max={float(fn.max()):.6g}"
            )
            print(
                "[PBC_DEBUG] |F_true| (eV/Å) stats: "
                f"min={float(fn.min())*hartree_to_ev:.6g}  p50={p50*hartree_to_ev:.6g}  "
                f"p90={p90*hartree_to_ev:.6g}  max={float(fn.max())*hartree_to_ev:.6g}"
            )
        else:
            print("[PBC_DEBUG] f_data present but empty/non-tensor?")
    else:
        print("[PBC_DEBUG] no f_data in batch")

    # ------------------- MIC distance check on ALL edges -------------------
    i = ind_2[:, 0]
    j = ind_2[:, 1]
    d = coord[j] - coord[i]  # (E,3)

    # Handle batched cell: (B,3,3) even for batch_size=1
    if cell.ndim == 3:
        assert cell.shape[0] == 1, f"[PBC_DEBUG] expected B=1 for debug, got cell.shape={tuple(cell.shape)}"
        H = cell[0]  # (3,3)
    elif cell.ndim == 2:
        H = cell
    else:
        raise AssertionError(f"[PBC_DEBUG] unexpected cell.ndim={cell.ndim}, cell.shape={tuple(cell.shape)}")

    invH = torch.linalg.inv(H)
    f = d @ invH
    f = f - torch.round(f)
    d_mic = f @ H
    dist = torch.linalg.norm(d_mic, dim=1)

    dist_np = dist.detach().cpu().numpy()
    print(
        "[PBC_DEBUG] MIC dist stats: "
        f"min={dist_np.min():.6f}  p50={np.quantile(dist_np,0.5):.6f}  "
        f"p90={np.quantile(dist_np,0.9):.6f}  max={dist_np.max():.6f}  (rc={rc})"
    )
    n_bad = int((dist > (rc + 1e-3)).sum().item())
    print(f"[PBC_DEBUG] edges with dist > rc+1e-3: {n_bad} / {dist.numel()}")

    coord_np = coord.detach().cpu().numpy()
    span = coord_np.max(axis=0) - coord_np.min(axis=0)
    H_np = H.detach().cpu().numpy()
    cell_lens = np.linalg.norm(H_np, axis=1)
    print("[PBC_DEBUG] coord span:", span)
    print("[PBC_DEBUG] cell lens :", cell_lens)
    print("[PBC_DEBUG] span/cell :", span / (cell_lens + 1e-12))

    # ------------------- Optional: |F_pred| magnitude via autograd -------------------
    if os.environ.get("WATER_TORCH_PBC_DEBUG_PRED", "0") == "1":
        model = model.to("cpu")
        model.eval()

        # Copy tensors; FORCE recompute diff/dist under grad by removing them if present
        tensors = dict(batch)
        tensors.pop("diff", None)
        tensors.pop("dist", None)

        coord_req = tensors["coord"].detach().clone().requires_grad_(True)
        tensors["coord"] = coord_req

        # --- TRAIN/EVAL semantics: label-space energy (Hartree-ish) ---
        if hasattr(model, "forward_train"):
            E_train = model.forward_train(tensors)  # (B,) in label space (scaled)
        else:
            # Fallback: if forward_train does not exist, use model(tensors) but then units depend on implementation
            E_train = model(tensors)

        if E_train.ndim == 2 and E_train.shape[1] == 1:
            E_train = E_train[:, 0]

        dE = torch.autograd.grad(E_train.sum(), coord_req, create_graph=False, retain_graph=False)[0]
        F_pred_train = -dE  # label-space forces (scaled label space)
        mag = torch.linalg.norm(F_pred_train, dim=-1).detach().cpu().numpy()

        print(
            "[PBC_DEBUG] |F_pred_train| (label units; Hartree/Å if e_scale=1) stats: "
            f"min={mag.min():.6g}  p50={np.quantile(mag,0.5):.6g}  "
            f"p90={np.quantile(mag,0.9):.6g}  max={mag.max():.6g}"
        )
        print(
            "[PBC_DEBUG] |F_pred_train| (approx eV/Å) stats: "
            f"min={mag.min()*hartree_to_ev:.6g}  p50={np.quantile(mag,0.5)*hartree_to_ev:.6g}  "
            f"p90={np.quantile(mag,0.9)*hartree_to_ev:.6g}  max={mag.max()*hartree_to_ev:.6g}"
        )

        # --- Optional: also show PREDICT semantics (eV energy -> eV/Å forces) ---
        if hasattr(model, "forward") and hasattr(model, "forward_train"):
            coord_req2 = tensors["coord"].detach().clone().requires_grad_(True)
            tensors2 = dict(tensors)
            tensors2["coord"] = coord_req2
            E_out = model(tensors2)  # predict energy in eV
            if E_out.ndim == 2 and E_out.shape[1] == 1:
                E_out = E_out[:, 0]
            dE2 = torch.autograd.grad(E_out.sum(), coord_req2, create_graph=False, retain_graph=False)[0]
            F_pred_out = -dE2  # eV/Å
            mag2 = torch.linalg.norm(F_pred_out, dim=-1).detach().cpu().numpy()
            print(
                "[PBC_DEBUG] |F_pred_out| (eV/Å) stats: "
                f"min={mag2.min():.6g}  p50={np.quantile(mag2,0.5):.6g}  "
                f"p90={np.quantile(mag2,0.9):.6g}  max={mag2.max():.6g}"
            )


HARTREE_TO_EV = 27.2114

def timed_iterator(it, name: str):
    """
    Wrap an iterator so that each `next()` call is timed.

    This measures:
      - neighbor list construction
      - diff/dist
      - batching
      - cache hits/misses

    Zero behavior change.
    """
    enabled = os.environ.get("WATER_TORCH_PROFILE", "0") == "1"
    while True:
        with walltime(name, enabled=enabled):
            yield next(it)

class TimedModel:
    """
    Thin wrapper to time model forward calls.
    """
    def __init__(self, model, name="model_forward"):
        self.model = model
        self.name = name
        self.enabled = os.environ.get("WATER_TORCH_PROFILE", "0") == "1"

    def __call__(self, *args, **kwargs):
        with walltime(self.name, enabled=self.enabled):
            return self.model(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.model, attr)

def _count_runner_frames(path: Path) -> int:
    """Count structures in a RuNNer input.data file by counting 'begin' markers (streaming, low RAM)."""
    n = 0
    with path.open("rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.lstrip().lower().startswith("begin"):
                n += 1
    return n

def _print_rmse(*, tag: str, e_rmse_pw_ha: float, f_rmse_ha_per_a: float, n_water: int) -> None:
    """Print energy/force RMSE in both Hartree and eV units."""
    e_ev = e_rmse_pw_ha * HARTREE_TO_EV
    f_ev_a = f_rmse_ha_per_a * HARTREE_TO_EV
    print(
        f"{tag} energy RMSE per water: {e_rmse_pw_ha:.6g} Hartree  |  {e_ev:.6g} eV/H2O"
        f"  (N_WATER={n_water})"
    )
    print(
        f"{tag} force  RMSE per comp: {f_rmse_ha_per_a:.6g} Hartree/Å | {f_ev_a:.6g} eV/Å"
    )

def _find_water_input_data() -> Path:
    """Resolve the local RuNNer dataset path (input.data)."""
    env = os.environ.get("WATER_INPUT_DATA", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"WATER_INPUT_DATA points to missing file: {p}")

    for cand in (Path.cwd() / "input.data", Path(__file__).resolve().parent / "input.data"):
        if cand.exists():
            return cand.resolve()

    raise FileNotFoundError(
        "Could not find 'input.data'. Set env WATER_INPUT_DATA=/path/to/input.data "
        "or place input.data in the working directory."
    )



def _zero_baseline_rmse_per_water(eval_ds, *, n_water: int, use_force: bool) -> Tuple[float, float]:
    """Compute (energy_rmse_per_water, force_rmse_per_component) for the zero predictor in LABEL space.

    LABEL space for water RuNNer:
      - energies: Hartree (per frame)  -> we convert to per-water Hartree by / n_water
      - forces:   Hartree/Å
    """
    if n_water <= 0:
        raise ValueError(f"n_water must be positive, got {n_water}")

    e_sq_sum = 0.0
    e_count = 0
    f_sq_sum = 0.0
    f_count = 0

    for ex in eval_ds:
        e_true_total = float(ex["e_data"].numpy().astype(np.float64))  # Hartree
        e_true_pw = e_true_total / float(n_water)                      # Hartree per water
        err_e = 0.0 - e_true_pw                                        # label space
        e_sq_sum += err_e * err_e
        e_count += 1

        if use_force:
            f_true = ex["f_data"].numpy().astype(np.float64)           # Hartree/Å
            f_sq_sum += (f_true * f_true).sum()
            f_count += f_true.size

    e_rmse = float(np.sqrt(e_sq_sum / max(e_count, 1)))
    f_rmse = float(np.sqrt(f_sq_sum / max(f_count, 1)))
    return e_rmse, f_rmse

def _torch_eval_mae_from_yml_per_water(
    *,
    model,
    params: dict,
    eval_yml: Path,
    scratch_dir: Path,
    device: str,
    eval_steps: int,
    batch_size_eval: int,
    n_water: int,
) -> Tuple[float, float]:
    """
    Compute (energy_mae_per_water, force_mae_per_component) on eval set using the Torch YAML pipeline.

    Energy error definition:
      E_err = (E_pred/e_unit) - E_true   (same as torch runtime convention)

    Then convert *both* predicted and true energies to per-water:
      E_pred_pw = E_pred_total / n_water
      E_true_pw = E_true_total / n_water

    Forces:
      F_pred = -dE/dR (autograd), compare with F_true in the same convention:
      F_err = (F_pred/e_unit) - F_true
    """
    if n_water <= 0:
        raise ValueError(f"n_water must be positive, got {n_water}")

    from pinn.io import load_tfrecord
    from pinn.io.base import sparse_batch
    from pinn.torch.runtime import TorchDataOptions, TorchRuntime

    mp = params.get("model", {}).get("params", {}) or {}
    e_unit = float(mp.get("e_unit", 1.0))
    use_force = bool(mp.get("use_force", True))

    ds = load_tfrecord(str(eval_yml), shuffle=False)
    ds = ds.apply(sparse_batch(int(batch_size_eval), drop_remainder=False))

    opts = TorchDataOptions(
        batch_size=int(batch_size_eval),
        shuffle=False,
        preprocess=True,
        cache=True,
        cache_ram=True,
        scratch_dir=str(scratch_dir),
        device=device,
    )
    rt = TorchRuntime(model=model, params=params, options=opts)

    e_abs_sum = 0.0
    e_count = 0
    f_abs_sum = 0.0
    f_count = 0

    it = iter(ds)
    for _ in range(int(eval_steps)):
        try:
            batch = next(it)
        except StopIteration:
            break

        tensors = rt.to_torch(batch)


        coord = tensors["coord"].detach().clone().requires_grad_(True)
        elems = tensors["elems"]
        ind_1 = tensors.get("ind_1", None)
        cell = tensors.get("cell", None)
        shift = tensors.get("shift", None)

        out = rt.forward(
            coord=coord,
            elems=elems,
            ind_1=ind_1,
            cell=cell,
            shift=shift,
        )
        E_pred_total = out["energy"]  # (B,) typically
        E_true_total = tensors["e_data"]  # (B,)

        # Convert to per-water energies (vectorized across batch).
        denom = float(n_water)
        E_pred_pw = E_pred_total / denom
        E_true_pw = E_true_total / denom

        e_err = (E_pred_pw / e_unit) - E_true_pw
        e_abs_sum += torch.abs(e_err).sum().item()
        e_count += int(E_true_pw.numel())

        if use_force:
            dE_dR = torch.autograd.grad(E_pred_total.sum(), coord, create_graph=False, retain_graph=False)[0]
            F_pred = -dE_dR
            F_true = tensors["f_data"]
            f_err = (F_pred / e_unit) - F_true
            f_abs_sum += torch.abs(f_err).sum().item()
            f_count += int(F_true.numel())

    e_mae = e_abs_sum / max(e_count, 1)
    f_mae = f_abs_sum / max(f_count, 1)
    return float(e_mae), float(f_mae)


@pytest.mark.slow
def test_water_runner_torch_pinet2_beats_zero_baseline(tmp_path: Path, monkeypatch) -> None:
    """Regression test for torch PiNet2 on liquid water RuNNer dataset."""
    monkeypatch.setenv("PINN_BACKEND", "torch")

    data_path = _find_water_input_data()

    # --------- dataset size (frames) ----------
    n_total = _count_runner_frames(data_path)
    print(f"RuNNer input.data frames detected: {n_total}")

    # --------- knobs ----------
    seed = int(os.environ.get("WATER_TORCH_SHUFFLE_SEED", "1"))

    num_train_steps = int(os.environ.get("WATER_TORCH_MAX_STEPS", "2000"))
    eval_steps = int(os.environ.get("WATER_TORCH_EVAL_STEPS", "200"))

    batch_size_train = int(os.environ.get("WATER_TORCH_BATCH_TRAIN", "1"))
    batch_size_eval = int(os.environ.get("WATER_TORCH_BATCH_EVAL", "1"))

    beat_frac = float(os.environ.get("WATER_TORCH_BEAT_BASELINE_FRAC", "0.02"))
    n_water = int(os.environ.get("WATER_TORCH_N_WATER", "64"))

    # --------- load + split ----------
    splits = {"train": 8, "eval": 2}
    ds_splits = load_runner(str(data_path), splits=splits, shuffle=True, seed=seed)
    train_ds = ds_splits["train"]   # use full 8/10 split
    eval_ds  = ds_splits["eval"]    # use full 2/10 split

    # --------- model params ----------
    params = {
        "model": {
            "name": "potential_model",
            "params": {
                "use_force": True,
                "use_e_per_atom": False,
                "log_e_per_atom": False,  # we care per-water instead
                "e_scale": 1.0,
                "e_unit": 27.2114,
                "e_loss_multiplier": 10.0,
                "f_loss_multiplier": 100.0,
            },
        },
        "network": {
            "name": "PiNet2",
            "params": {
                "atom_types": [1, 8],
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
                        "initial_learning_rate": 5.0e-05,
                        "decay_steps": 100000,
                        "decay_rate": 0.994,
                    },
                },
            },
        },
        "model_dir": str(tmp_path / "pinet2_water_torch"),
    }

    ret = init_params(params, dataset=train_ds)
    if ret is not None:
        params = ret

    # --------- write TFRecord YAMLs ----------
    train_yml = tmp_path / "water_train.yml"
    eval_yml = tmp_path / "water_eval.yml"
    write_tfrecord(str(train_yml), train_ds)
    write_tfrecord(str(eval_yml), eval_ds)
    

    # --------- baseline (per-water energy) ----------
    e_scale = float(params["model"]["params"].get("e_scale", 1.0))
    e0_raw, f0_raw = _zero_baseline_rmse_per_water(eval_ds, n_water=n_water, use_force=True)
    # baseline in scaled-label space (what runtime reports)
    _print_rmse(tag="Zero baseline", e_rmse_pw_ha=e0_raw, f_rmse_ha_per_a=f0_raw, n_water=n_water)

    # --------- train (torch backend) ----------
    scratch_dir = tmp_path / "scratch"
    model = TimedModel(pinn.get_model(params))
    if os.environ.get("WATER_TORCH_PBC_DEBUG", "0") == "1":
        debug_one_batch_pbc_from_yml(yml_path=train_yml, params=params, scratch_dir=scratch_dir)

    # --------- coarse profiling (optional) ----------
    if os.environ.get("WATER_TORCH_PROFILE", "0") == "1":
        import pinn.torch.runtime as rt

        _orig_repeat = rt._repeat_batches_from_built_examples

        def _timed_repeat(*args, **kwargs):
            it = _orig_repeat(*args, **kwargs)
            return timed_iterator(it, "train_it.next")

        rt._repeat_batches_from_built_examples = _timed_repeat

    metrics = pinn.train_and_evaluate(
        model=model,
        params=params,
        data={},  # IMPORTANT: avoid LJ iter_batches path (expects coord/elems/e_data/f_data)
        train_yml=str(train_yml),
        eval_yml=str(eval_yml),
        max_steps=int(num_train_steps),
        eval_steps=int(eval_steps),
        batch_size_train=int(batch_size_train),
        batch_size_eval=int(batch_size_eval),
        shuffle_buffer=0,
        preprocess=True,
        cache=True,
        cache_ram=True,
        scratch_dir=str(scratch_dir),
    )
    print("torch train_and_evaluate metrics:", metrics)

    e_scale = float(params["model"]["params"].get("e_scale", 1.0))

    # Runtime metrics should be in SCALED label space if you now match TF semantics.
    e_rmse_scaled = float(metrics["METRICS/E_RMSE"])
    f_rmse_scaled = float(metrics["METRICS/F_RMSE"])

    # Convert back to RAW label space (Hartree and Hartree/Å for water labels)
    e_rmse_raw = e_rmse_scaled / e_scale
    f_rmse_raw = f_rmse_scaled / e_scale

    # Convert energy RMSE to per-water
    e_rmse_per_water = e_rmse_raw / float(n_water)

    _print_rmse(tag="Final", e_rmse_pw_ha=e_rmse_per_water, f_rmse_ha_per_a=f_rmse_raw, n_water=n_water)

    # --------- assert: beat baseline (RMSE-to-RMSE in label space) ----------
    assert np.isfinite(e_rmse_per_water) and np.isfinite(f_rmse_raw)

    assert e_rmse_per_water < (1.0 - beat_frac) * e0_raw, (
        f"Energy RMSE per water {e_rmse_per_water} did not beat baseline {e0_raw} by {beat_frac:.0%}"
    )
    assert f_rmse_raw < (1.0 - beat_frac) * f0_raw, (
        f"Force RMSE {f_rmse_raw} did not beat baseline {f0_raw} by {beat_frac:.0%}"
    )