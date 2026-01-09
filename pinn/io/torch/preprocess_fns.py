# pinn/io/torch/preprocess_fns.py
"""
Shared preprocessing functions for Torch backend.

These functions are the single source of truth for:
- building neighbor list (ind_2, shift) using a CellList-style builder
- computing diff/dist from coord, ind_2, cell, shift, ind_1 (supports per-structure cell)
- building prop (one-hot) from elems

Design:
- Used by IO preprocessing (build_dataset / dataloader path)
- Used by model-side PreprocessLayerTorch (compatibility / safety)

Keep this file free of model imports to avoid circular dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, List

import torch
import torch.nn as nn


from typing import Dict, Optional, List

import torch
import torch.nn as nn


@torch.no_grad()
def build_nl_celllist(
    *,
    ind_1: torch.Tensor,
    coord: torch.Tensor,
    cell: Optional[torch.Tensor],
    rc: float,
    nl_builder: nn.Module,
) -> Dict[str, torch.Tensor]:
    """
    Build a directed neighbor list for a *batched* structure set.

    Change 2 integrated:
      - If the NL builder returns "diff" and/or "dist", propagate them so callers
        can skip compute_diff_dist(...) and avoid paying twice.

    Returns dict with at least:
      - ind_2: (E,2) long
      - shift: (E,3) long
    Optionally (if provided by builder):
      - diff: (E,3) float (same dtype/device as coord)
      - dist: (E,)  float (same dtype/device as coord)
    """
    if ind_1.dtype != torch.long:
        ind_1 = ind_1.long()
    batch = ind_1[:, 0] if ind_1.ndim == 2 else ind_1  # (N,)

    device = coord.device
    if batch.numel() == 0:
        return {
            "ind_2": coord.new_zeros((0, 2), dtype=torch.long),
            "shift": coord.new_zeros((0, 3), dtype=torch.long),
        }

    # ---------- VECTORIZED FAST PATH ----------
    # If nl_builder supports a batched signature, use it and return immediately.
    # Expected batched signature:
    #   nl_builder(coord, ind_1=ind_1, cell=cell, rc=rc)  (rc optional)
    # or:
    #   nl_builder(coord, ind_1=ind_1, cell=cell)
    try:
        try:
            nl = nl_builder(coord, ind_1=ind_1, cell=cell, rc=rc)
        except TypeError:
            nl = nl_builder(coord, ind_1=ind_1, cell=cell)

        if isinstance(nl, dict) and "ind_2" in nl:
            ind_2 = nl["ind_2"].to(dtype=torch.long, device=device)

            shift = nl.get("shift", None)
            if shift is None:
                shift = coord.new_zeros((ind_2.shape[0], 3), dtype=torch.long)
            else:
                shift = shift.to(dtype=torch.long, device=device)

            out: Dict[str, torch.Tensor] = {"ind_2": ind_2, "shift": shift}

            # --- Change 2: propagate diff/dist if builder already computed them ---
            if "diff" in nl:
                out["diff"] = nl["diff"].to(device=device, dtype=coord.dtype)
            if "dist" in nl:
                out["dist"] = nl["dist"].to(device=device, dtype=coord.dtype)

            return out
    except TypeError:
        # Old-style builder signature, fall back to per-structure path below
        pass

    # Contiguous blocks => nondecreasing batch ids
    is_nondecreasing = bool((batch[1:] >= batch[:-1]).all().item()) if batch.numel() > 1 else True
    cell_is_per_struct = (cell is not None and cell.ndim == 3)

    # ---------- FAST PATH: block-contiguous ids via unique_consecutive ----------
    if is_nondecreasing:
        uniq, counts = torch.unique_consecutive(batch, return_counts=True)

        starts = torch.zeros((counts.numel() + 1,), dtype=torch.long, device=device)
        starts[1:] = torch.cumsum(counts.to(device=device, dtype=torch.long), dim=0)

        ind2_list: List[torch.Tensor] = []
        shift_list: List[torch.Tensor] = []
        diff_list: List[torch.Tensor] = []
        dist_list: List[torch.Tensor] = []

        for k in range(uniq.numel()):
            s0 = int(starts[k].item())
            s1 = int(starts[k + 1].item())
            if s1 <= s0:
                continue

            coord_b = coord[s0:s1]

            if cell is None:
                nl_b = nl_builder(coord_b, cell=None)
            else:
                # If cell is per-structure (n_struct,3,3), we assume it aligns with batch order (k)
                H = cell[k] if cell_is_per_struct else cell
                nl_b = nl_builder(coord_b, cell=H)

            ind_2_local = nl_b.get("ind_2", None)
            if ind_2_local is None or ind_2_local.numel() == 0:
                continue

            gi = ind_2_local[:, 0] + s0
            gj = ind_2_local[:, 1] + s0
            ind2_list.append(torch.stack([gi, gj], dim=1))

            if "shift" in nl_b:
                shift_list.append(nl_b["shift"].to(dtype=torch.long, device=device))
            else:
                shift_list.append(torch.zeros((ind_2_local.shape[0], 3), dtype=torch.long, device=device))

            # --- Change 2: propagate diff/dist if present ---
            if "diff" in nl_b:
                diff_list.append(nl_b["diff"].to(device=device, dtype=coord.dtype))
            if "dist" in nl_b:
                dist_list.append(nl_b["dist"].to(device=device, dtype=coord.dtype))

        if not ind2_list:
            return {
                "ind_2": coord.new_zeros((0, 2), dtype=torch.long),
                "shift": coord.new_zeros((0, 3), dtype=torch.long),
            }

        out: Dict[str, torch.Tensor] = {
            "ind_2": torch.cat(ind2_list, dim=0),
            "shift": torch.cat(shift_list, dim=0),
        }
        if diff_list:
            out["diff"] = torch.cat(diff_list, dim=0)
        if dist_list:
            out["dist"] = torch.cat(dist_list, dim=0)
        return out

    # ---------- FALLBACK: mask-based ----------
    ind2_list: List[torch.Tensor] = []
    shift_list: List[torch.Tensor] = []
    diff_list: List[torch.Tensor] = []
    dist_list: List[torch.Tensor] = []

    for b in batch.unique(sorted=True).tolist():
        idx = (batch == b).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue

        coord_b = coord[idx]
        if cell is None:
            nl_b = nl_builder(coord_b, cell=None)
        else:
            H = cell[b] if cell_is_per_struct else cell
            nl_b = nl_builder(coord_b, cell=H)

        ind_2_local = nl_b.get("ind_2", None)
        if ind_2_local is None or ind_2_local.numel() == 0:
            continue

        gi = idx[ind_2_local[:, 0]]
        gj = idx[ind_2_local[:, 1]]
        ind2_list.append(torch.stack([gi, gj], dim=1))

        if "shift" in nl_b:
            shift_list.append(nl_b["shift"].to(dtype=torch.long, device=device))
        else:
            shift_list.append(torch.zeros((ind_2_local.shape[0], 3), dtype=torch.long, device=device))

        # --- Change 2: propagate diff/dist if present ---
        if "diff" in nl_b:
            diff_list.append(nl_b["diff"].to(device=device, dtype=coord.dtype))
        if "dist" in nl_b:
            dist_list.append(nl_b["dist"].to(device=device, dtype=coord.dtype))

    if not ind2_list:
        return {
            "ind_2": coord.new_zeros((0, 2), dtype=torch.long),
            "shift": coord.new_zeros((0, 3), dtype=torch.long),
        }

    out = {
        "ind_2": torch.cat(ind2_list, dim=0),
        "shift": torch.cat(shift_list, dim=0),
    }
    if diff_list:
        out["diff"] = torch.cat(diff_list, dim=0)
    if dist_list:
        out["dist"] = torch.cat(dist_list, dim=0)
    return out



def compute_diff_dist(
    *,
    coord: torch.Tensor,
    ind_2: torch.Tensor,
    cell: Optional[torch.Tensor],
    shift: Optional[torch.Tensor],
    ind_1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute diff and dist for directed edges.

    Args:
        coord: (N,3)
        ind_2: (E,2) long
        cell: None, (3,3), or (B,3,3)
        shift: None or (E,3) long integer image shifts
        ind_1: (N,1)/(N,2) or (N,) used for structure id mapping if cell is per-structure

    Returns:
        diff: (E,3)
        dist: (E,)
    """
    i = ind_2[:, 0]
    j = ind_2[:, 1]

    # No PBC info
    if cell is None or shift is None:
        diff = coord[j] - coord[i]
        dist = torch.linalg.norm(diff, dim=1)
        return diff, dist

    # PBC global cell
    if cell.ndim == 2:
        H = cell.to(device=coord.device, dtype=coord.dtype)     # (3,3)
        t = shift.to(coord.dtype) @ H                           # (E,3)
        diff = (coord[j] + t) - coord[i]
        dist = torch.linalg.norm(diff, dim=1)
        return diff, dist

    # PBC per-structure cell
    if cell.ndim == 3:
        if ind_1.dtype != torch.long:
            ind_1 = ind_1.long()
        batch = ind_1[:, 0] if ind_1.ndim == 2 else ind_1       # (N,)
        sid = batch[i]                                          # (E,)
        H_pair = cell[sid].to(device=coord.device, dtype=coord.dtype)  # (E,3,3)
        t = torch.einsum("ei,eij->ej", shift.to(coord.dtype), H_pair)  # (E,3)
        diff = (coord[j] + t) - coord[i]
        dist = torch.linalg.norm(diff, dim=1)
        return diff, dist

    raise ValueError(f"Unexpected cell shape {tuple(cell.shape)}")


def atomic_onehot(elems: torch.Tensor, atom_types: Sequence[int]) -> torch.Tensor:
    """
    One-hot encode atomic numbers.

    Args:
        elems: (N,) integer atomic numbers
        atom_types: list of allowed atomic numbers defining channel order

    Returns:
        prop: (N, len(atom_types)) float tensor with 0/1 entries
    """
    if elems.dtype != torch.long:
        elems = elems.long()
    device = elems.device
    types = torch.tensor(list(atom_types), dtype=torch.long, device=device)  # (T,)
    # prop[n,t] = (elems[n] == types[t])
    prop = (elems[:, None] == types[None, :]).to(dtype=torch.float32)
    return prop
