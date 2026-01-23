"""
pbc_sanity_check.py

Compare PiNNch Torch PBC neighbor geometry (ind_2/shift + diff/dist)
against ASE neighbor_list for a single structure.

This tests the real thing:
  - loads ONE structure from your actual TFRecord YAML (coord/elems/cell)
  - builds ASE reference neighbor list under PBC
  - builds Torch neighbor list using PiNNch's actual Torch cell-list builder:
        pinn.networks.pinet_torch.CellListNLPyTorch
  - compares edge counts and distance quantiles (+ optional strict multiset match)

NEW: Debug output (enabled by default) that prints:
  - shift statistics (min/max per axis, unique values sample)
  - diff norm min/median/max
  - the single worst (largest dist) pair: indices, shift, diff vector, and cell

Notes:
- TensorFlow logs on macOS may appear because TF is used to read TFRecords.
  That does NOT mean we are testing TF training. We test Torch PBC geometry.
"""

from __future__ import annotations

import numpy as np


def run_pbc_sanity_check(
    yml_path: str,
    rc: float,
    atom_types=(1, 8),
    atol: float = 1e-6,
    rtol: float = 1e-6,
    debug: bool = True,
    debug_unique_limit: int = 30,
):
    """
    Run one-structure PBC geometry sanity check.

    Args:
        yml_path: TFRecord YAML used by your water system (must include 'cell').
        rc: Cutoff radius in Angstrom; MUST match training.
        atom_types: Atomic numbers present (water: (1, 8)).
        atol: Absolute tolerance for strict distance multiset match.
        rtol: Relative tolerance for strict distance multiset match.
        debug: If True, print detailed debug stats for shift/diff/dist.
        debug_unique_limit: Max number of unique shift scalar values to print.

    Prints:
        Edge counts and distance quantiles for ASE vs Torch.
        If debug=True, additional diagnostics (see module docstring).
    """
    import torch
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    from pinn.io import load_tfrecord
    from pinn.io.build_dataset import iter_numpy_examples
    from pinn.io.torch.preprocess import preprocess_batch_torch

    # In PiNNch.zip, CellListNLPyTorch is defined here:
    from pinn.networks.pinet_torch import CellListNLPyTorch

    # --- load one example (numpy dict) ---
    ds = load_tfrecord(yml_path, shuffle=False)
    ex = next(iter(iter_numpy_examples(ds)))

    assert "coord" in ex and "elems" in ex, "Example missing coord/elems."
    assert "cell" in ex and ex["cell"] is not None, (
        "Example missing cell => cannot test PBC. "
        "If unexpected, your dataset/batching pipeline is dropping 'cell'."
    )

    coord = np.array(ex["coord"], dtype=np.float64)
    elems = np.array(ex["elems"], dtype=np.int32)
    cell = np.array(ex["cell"], dtype=np.float64)

    # --- build ASE truth ---
    atoms = Atoms(numbers=elems, positions=coord, cell=cell, pbc=True)

    # ASE returns i, j, S where displacement is r_j + S @ cell - r_i
    i_ase, j_ase, S_ase = neighbor_list("ijS", atoms, cutoff=float(rc), self_interaction=False)
    diff_ase = atoms.positions[j_ase] + (S_ase @ atoms.cell.array) - atoms.positions[i_ase]
    dist_ase = np.linalg.norm(diff_ase, axis=1)

    # --- build Torch batch and preprocess (ind_2/shift/diff/dist) ---
    t = {
        "coord": torch.tensor(coord, dtype=torch.float32),
        "elems": torch.tensor(elems, dtype=torch.long),
        "cell": torch.tensor(cell, dtype=torch.float32),
        # Some preprocess paths expect this key to exist
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

    # Pull torch outputs
    # (keys expected when make_diff_dist=True)
    ind_2_t = t.get("ind_2", None)
    shift_t = t.get("shift", None)
    diff_t = t.get("diff", None)
    dist_t = t.get("dist", None)

    assert dist_t is not None, "Torch preprocess did not produce 'dist'."
    dist_t_np = dist_t.detach().cpu().numpy()

    # --- compare counts ---
    print(f"ASE edges:   {len(dist_ase)}")
    print(f"Torch edges: {len(dist_t_np)}")

    # --- compare distance distributions (robust even if ordering differs) ---
    dist_ase_sorted = np.sort(dist_ase)
    dist_t_sorted = np.sort(dist_t_np)

    m = min(len(dist_ase_sorted), len(dist_t_sorted))
    if m == 0:
        raise RuntimeError("No neighbors found; check rc and input.")

    for q in (0.0, 0.1, 0.5, 0.9, 1.0):
        da = np.quantile(dist_ase_sorted, q)
        dt = np.quantile(dist_t_sorted, q)
        print(f"q={q:>3}: ASE {da: .6f}  Torch {dt: .6f}  diff {dt-da: .3e}")

    # Optional strict check: same multiset of distances within tol
    if len(dist_ase_sorted) == len(dist_t_sorted):
        ok = np.allclose(dist_ase_sorted, dist_t_sorted, atol=atol, rtol=rtol)
        print("Exact distance multiset match:", ok)
        if not ok:
            idx = int(np.argmax(np.abs(dist_ase_sorted - dist_t_sorted)))
            print(
                "Worst mismatch at idx",
                idx,
                "ASE",
                float(dist_ase_sorted[idx]),
                "Torch",
                float(dist_t_sorted[idx]),
            )
    else:
        print("Edge counts differ => neighbor list mismatch (missing PBC neighbors or duplicates).")

    # --- debug: shift/diff diagnostics + worst pair dump ---
    if debug:
        print("\n--- DEBUG: Torch shift/diff/dist diagnostics ---")

        if shift_t is None:
            print("Torch output is missing 'shift' key.")
        else:
            shift_np = shift_t.detach().cpu().numpy()
            print("shift dtype:", shift_np.dtype, "shape:", shift_np.shape)
            if shift_np.size > 0:
                try:
                    mn = shift_np.min(axis=0)
                    mx = shift_np.max(axis=0)
                    print("shift min per axis:", mn, "shift max per axis:", mx)
                except Exception:
                    print("shift min/max: (could not compute per-axis)")

                u = np.unique(shift_np.reshape(-1))
                print(f"unique shift scalar values (showing up to {debug_unique_limit}):", u[:debug_unique_limit])
                if u.size > debug_unique_limit:
                    print(f"... ({u.size - debug_unique_limit} more)")

        if diff_t is None:
            print("Torch output is missing 'diff' key.")
        else:
            diff_np = diff_t.detach().cpu().numpy()
            dn = np.linalg.norm(diff_np, axis=1) if diff_np.size else np.array([])
            if dn.size:
                print(
                    "diff norm min/median/max:",
                    float(dn.min()),
                    float(np.median(dn)),
                    float(dn.max()),
                )
            else:
                print("diff norm stats: (empty)")

        # Worst distance pair details
        k = int(np.argmax(dist_t_np))
        print("worst idx (Torch dist):", k, "dist:", float(dist_t_np[k]))

        if ind_2_t is not None:
            ind2_np = ind_2_t.detach().cpu().numpy()
            if len(ind2_np) > k:
                i, j = int(ind2_np[k, 0]), int(ind2_np[k, 1])
                print("worst pair (i, j):", (i, j))
                print("coord[i]:", coord[i])
                print("coord[j]:", coord[j])

        if shift_t is not None and shift_t.numel() > 0:
            shift_np = shift_t.detach().cpu().numpy()
            if len(shift_np) > k:
                print("worst shift:", shift_np[k])

        if diff_t is not None and diff_t.numel() > 0:
            diff_np = diff_t.detach().cpu().numpy()
            if len(diff_np) > k:
                print("worst diff:", diff_np[k])

        print("cell (from example):\n", cell)


if __name__ == "__main__":
    run_pbc_sanity_check(
        yml_path="/private/var/folders/5_/10m7kj3n6gl4944kpt_php900000gn/T/pytest-of-chazh245/pytest-471/test_water_runner_torch_pinet20/water_train.yml",
        rc=4.5,
        atom_types=(1, 8),
        debug=True,
    )