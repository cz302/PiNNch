# pinn/io/torch/preprocess.py
"""
Torch IO preprocessing entry point.

This module provides a batch-level preprocessing function that mirrors the
behavior of PreprocessLayerTorch, but lives in IO (so models don't have to
recompute geometry features).
"""

from __future__ import annotations

from typing import Dict, Sequence, Any, Optional

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
    """
    Preprocess a single-structure Torch batch.

    Adds/ensures:
      - elems (alias of z if needed)
      - prop (one-hot of elems)
      - ind_2 (+ shift for PBC) via build_nl_celllist if missing
      - diff/dist computed by compute_diff_dist() (single source of truth)

    PBC convention:
      - If shift is present, diff = (r_j - r_i) + (shift @ cell)  (multi-image safe)
      - If shift is absent, compute_diff_dist infers a compatible shift by rounding.
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
        out.update(nl)  # keeps ind_2 and shift

        # IMPORTANT:
        # If NL builder gave diff/dist, drop them because builder runs under no_grad
        # and we need autograd-connected diff/dist for forces/stress.
        out.pop("diff", None)
        out.pop("dist", None)

    if make_diff_dist and (("diff" not in out) or ("dist" not in out)):
        diff, dist = compute_diff_dist(
            coord=out["coord"],
            ind_2=out["ind_2"],
            cell=out.get("cell", None),
            shift=out.get("shift", None),  # <-- critical: use shift if provided
            ind_1=out["ind_1"],
        )
        out["diff"] = diff
        out["dist"] = dist

    return out