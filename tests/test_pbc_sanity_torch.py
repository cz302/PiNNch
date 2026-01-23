# -*- coding: utf-8 -*-
"""
Pytest: Torch PBC neighbor geometry sanity check vs ASE for one RuNNer water structure.

Assumptions:
- water_train.yml and water_train.tfr are in the same folder as this pytest file.
- We test Torch neighborlist geometry (ind_2/shift + diff/dist) against ASE neighbor_list.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


HARTREE_TO_EV = 27.2114  # not used here but fine to keep around


def _load_one_example_from_yml(yml_path: Path) -> dict:
    """Load ONE structure as a numpy dict from TFRecord YAML."""
    from pinn.io import load_tfrecord
    from pinn.io.build_dataset import iter_numpy_examples

    ds = load_tfrecord(str(yml_path), shuffle=False)
    return next(iter(iter_numpy_examples(ds)))


def _torch_preprocess_one(
    *,
    coord: np.ndarray,
    elems: np.ndarray,
    cell: np.ndarray,
    rc: float,
    atom_types=(1, 8),
):
    """Run the real Torch preprocess that produces ind_2/shift/diff/dist."""
    import torch
    from pinn.io.torch.preprocess import preprocess_batch_torch
    from pinn.networks.pinet_torch import CellListNLPyTorch

    t = {
        "coord": torch.tensor(coord, dtype=torch.float32),
        "elems": torch.tensor(elems, dtype=torch.long),
        "cell": torch.tensor(cell, dtype=torch.float32),
        # some preprocess paths expect ind_1 to exist
        "ind_1": torch.zeros((coord.shape[0], 1), dtype=torch.long),
    }

    nl_builder = CellListNLPyTorch(float(rc))

    t = preprocess_batch_torch(
        t,
        atom_types=list(atom_types),
        rc=float(rc),
        nl_builder=nl_builder,
        make_diff_dist=True,
    )
    return t


def _ase_neighbor_dists(coord: np.ndarray, elems: np.ndarray, cell: np.ndarray, rc: float) -> np.ndarray:
    """ASE reference neighbor distances under PBC."""
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    atoms = Atoms(numbers=elems, positions=coord, cell=cell, pbc=True)

    # ASE returns i, j, S where displacement is r_j + S @ cell - r_i
    i_ase, j_ase, S_ase = neighbor_list("ijS", atoms, cutoff=float(rc), self_interaction=False)
    diff_ase = atoms.positions[j_ase] + (S_ase @ atoms.cell.array) - atoms.positions[i_ase]
    dist_ase = np.linalg.norm(diff_ase, axis=1)
    return dist_ase


def _debug_dump(*, coord, cell, dist_t_np, batch_torch, limit_unique=30) -> str:
    """Return a big debug string for failure messages."""
    ind_2_t = batch_torch.get("ind_2", None)
    shift_t = batch_torch.get("shift", None)
    diff_t = batch_torch.get("diff", None)

    lines = []
    lines.append("\n--- DEBUG: Torch shift/diff/dist diagnostics ---")

    if shift_t is None:
        lines.append("Torch output missing 'shift'.")
    else:
        shift_np = shift_t.detach().cpu().numpy()
        lines.append(f"shift dtype={shift_np.dtype} shape={shift_np.shape}")
        if shift_np.size > 0:
            try:
                mn = shift_np.min(axis=0)
                mx = shift_np.max(axis=0)
                lines.append(f"shift min per axis={mn} max per axis={mx}")
            except Exception:
                lines.append("shift min/max: could not compute per-axis")
            u = np.unique(shift_np.reshape(-1))
            lines.append(f"unique shift scalar values (up to {limit_unique}): {u[:limit_unique]}")
            if u.size > limit_unique:
                lines.append(f"... ({u.size - limit_unique} more)")

    if diff_t is None:
        lines.append("Torch output missing 'diff'.")
    else:
        diff_np = diff_t.detach().cpu().numpy()
        dn = np.linalg.norm(diff_np, axis=1) if diff_np.size else np.array([])
        if dn.size:
            lines.append(
                "diff norm min/median/max: "
                f"{float(dn.min())} {float(np.median(dn))} {float(dn.max())}"
            )
        else:
            lines.append("diff norm stats: empty")

    # worst distance pair details
    k = int(np.argmax(dist_t_np)) if dist_t_np.size else -1
    lines.append(f"worst idx (Torch dist)={k} dist={float(dist_t_np[k]) if k >= 0 else 'n/a'}")

    if ind_2_t is not None and k >= 0:
        ind2_np = ind_2_t.detach().cpu().numpy()
        if len(ind2_np) > k:
            i, j = int(ind2_np[k, 0]), int(ind2_np[k, 1])
            lines.append(f"worst pair (i,j)=({i},{j})")
            lines.append(f"coord[i]={coord[i]}")
            lines.append(f"coord[j]={coord[j]}")

    if shift_t is not None and shift_t.numel() > 0 and k >= 0:
        shift_np = shift_t.detach().cpu().numpy()
        if len(shift_np) > k:
            lines.append(f"worst shift={shift_np[k]}")

    if diff_t is not None and diff_t.numel() > 0 and k >= 0:
        diff_np = diff_t.detach().cpu().numpy()
        if len(diff_np) > k:
            lines.append(f"worst diff={diff_np[k]}")

    lines.append("cell (from example):\n" + str(cell))
    return "\n".join(lines)


@pytest.mark.parametrize("rc", [4.5])
def test_torch_pbc_neighborlist_matches_ase(rc: float) -> None:
    """
    Compare Torch CellListNLPyTorch-based neighbors vs ASE neighbor_list for ONE structure.

    Assertions:
    - same number of edges
    - sorted distance multiset matches within tolerances
    """
    here = Path(__file__).resolve().parent
    yml_path = here / "water_train.yml"
    assert yml_path.exists(), f"Missing {yml_path}. Put water_train.yml next to this test."

    ex = _load_one_example_from_yml(yml_path)

    assert "coord" in ex and "elems" in ex, "Example missing coord/elems."
    assert "cell" in ex and ex["cell"] is not None, (
        "Example missing cell => cannot test PBC. "
        "If unexpected, your dataset/batching pipeline is dropping 'cell'."
    )

    coord = np.array(ex["coord"], dtype=np.float64)
    elems = np.array(ex["elems"], dtype=np.int32)
    cell = np.array(ex["cell"], dtype=np.float64)

    dist_ase = _ase_neighbor_dists(coord, elems, cell, rc)

    t = _torch_preprocess_one(coord=coord, elems=elems, cell=cell, rc=rc, atom_types=(1, 8))
    dist_t = t.get("dist", None)
    assert dist_t is not None, "Torch preprocess did not produce 'dist'."
    dist_t_np = dist_t.detach().cpu().numpy().astype(np.float64, copy=False)

    # Basic sanity
    assert dist_ase.size > 0, "ASE found no neighbors; check rc/input."
    assert dist_t_np.size > 0, "Torch found no neighbors; check rc/input/preprocess."

    # Compare counts first
    if dist_ase.size != dist_t_np.size:
        msg = (
            f"Edge count mismatch: ASE={dist_ase.size} Torch={dist_t_np.size}\n"
            + _debug_dump(coord=coord, cell=cell, dist_t_np=dist_t_np, batch_torch=t)
        )
        raise AssertionError(msg)

    # Compare distance multisets (ordering-independent)
    dist_ase_sorted = np.sort(dist_ase)
    dist_t_sorted = np.sort(dist_t_np)

    # Tolerances: float32 pipeline + MIC rounding => keep modest but strict
    atol = 1e-5
    rtol = 1e-6

    ok = np.allclose(dist_ase_sorted, dist_t_sorted, atol=atol, rtol=rtol)
    if not ok:
        idx = int(np.argmax(np.abs(dist_ase_sorted - dist_t_sorted)))
        msg = (
            "Distance multiset mismatch.\n"
            f"worst idx={idx} ASE={float(dist_ase_sorted[idx])} Torch={float(dist_t_sorted[idx])} "
            f"absdiff={float(abs(dist_ase_sorted[idx]-dist_t_sorted[idx]))}\n"
        )
        # Add quantiles for quick glance
        for q in (0.0, 0.1, 0.5, 0.9, 1.0):
            da = float(np.quantile(dist_ase_sorted, q))
            dt = float(np.quantile(dist_t_sorted, q))
            msg += f"q={q:>3}: ASE {da:.6f} Torch {dt:.6f} diff {dt-da:.3e}\n"

        msg += _debug_dump(coord=coord, cell=cell, dist_t_np=dist_t_np, batch_torch=t)
        raise AssertionError(msg)
