# pinn/io/torch/preprocess.py
"""
Torch IO preprocessing entry point.

This module provides a batch-level preprocessing function that mirrors the
behavior of PreprocessLayerTorch, but lives in IO (so models don't have to
recompute geometry features).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Any

import torch
import torch.nn as nn

from pinn.io.torch.preprocess_fns import build_nl_celllist, compute_diff_dist, atomic_onehot


def preprocess_batch_torch(
    tensors: Dict[str, Any],
    *,
    atom_types: Sequence[int],
    rc: float,
    nl_builder: nn.Module,
    make_diff_dist: bool = True,
) -> Dict[str, Any]:
    """Preprocess a batched tensor dict for Torch backend.

    This function can optionally *skip* computing edge displacement vectors and
    distances (diff/dist). That is useful when you want to build the neighbor
    list (ind_2/shift) on CPU, but compute diff/dist later on GPU from the same
    coordinate tensor (to preserve gradients for forces).

    Args:
        tensors: Sparse-batch dict containing at least coord, ind_1, elems/z.
        atom_types: Atomic numbers included in the one-hot encoding.
        rc: Cutoff radius for neighbor list.
        nl_builder: Neighbor-list builder module.
        make_diff_dist: If False, only builds prop + (ind_2, shift). If True,
            also computes (diff, dist) if not already present.

    Returns:
        Updated sparse-batch dict.
    """
    out = dict(tensors)

    if "elems" not in out and "z" in out:
        out["elems"] = out["z"]

    if "prop" not in out:
        prop = atomic_onehot(out["elems"], atom_types).to(device=out["coord"].device)
        out["prop"] = prop.to(dtype=out["coord"].dtype)

    if "ind_2" not in out:
        cell = out.get("cell", None)
        nl = build_nl_celllist(
            ind_1=out["ind_1"],
            coord=out["coord"],
            cell=cell,
            rc=rc,
            nl_builder=nl_builder,
        )
        out.update(nl)

    if make_diff_dist and ("diff" not in out or "dist" not in out):
        # Only compute diff/dist when coord participates in autograd.
        # In runtime, coord.requires_grad_(True) is typically set later.
        if out["coord"].requires_grad:
            cell = out.get("cell", None)
            shift = out.get("shift", None)
            diff, dist = compute_diff_dist(
                coord=out["coord"],
                ind_2=out["ind_2"],
                cell=cell,
                shift=shift,
                ind_1=out["ind_1"],
            )
            out["diff"] = diff
            out["dist"] = dist

    return out