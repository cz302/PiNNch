# pinn/torch/celllist_batched.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class BatchedCellListConfig:
    rc: float
    cell_size_factor: float = 1.0  # cell_size = rc * factor (0.5 is a good default)
    max_pairs_per_atom: int = 0    # (unused here; keep for future safety caps)


def _get_box_lengths(cell: torch.Tensor) -> torch.Tensor:
    """
    Extract orthorhombic box lengths (Lx,Ly,Lz) from a (3,3) cell.
    Requires near-diagonal cell; otherwise raise.
    """
    if cell.shape != (3, 3):
        raise ValueError(f"Expected cell shape (3,3), got {tuple(cell.shape)}")

    offdiag = cell - torch.diag(torch.diagonal(cell))
    if torch.max(torch.abs(offdiag)).item() > 1e-6:
        raise NotImplementedError(
            "CellListNLBatched supports orthorhombic cells only (near-diagonal 3x3). "
            "Your cell has non-zero off-diagonal terms."
        )

    return torch.abs(torch.diagonal(cell))  # (3,)


class CellListNLBatched(nn.Module):
    """
    Batched *PBC-only* cell-list neighbor builder (orthorhombic).

    This module is intended to be used via NLRouter for PBC cases only.
    Non-PBC handling is intentionally removed.

    Inputs:
      coord: (N,3) float
      ind_1: (N,1) or (N,) long, structure id per atom (0..B-1 recommended)
      cell:
        - (3,3) global orthorhombic cell
        - (B,3,3) per-structure orthorhombic cell

    Returns:
      ind_2: (E,2) long directed edges i->j
      shift: (E,3) long integer image shifts in lattice-vector units
    """

    def __init__(self, cfg: BatchedCellListConfig):
        super().__init__()
        self.rc = float(cfg.rc)
        self.cell_size_factor = float(getattr(cfg, "cell_size_factor", 0.5))
        self.max_pairs_per_atom = int(getattr(cfg, "max_pairs_per_atom", 0))

        # Precompute stencil offsets for the chosen bin size.
        cell_size = max(1e-6, self.rc * self.cell_size_factor)
        # Need to search neighbor cells out to ceil(rc / cell_size) in each direction
        R = int(torch.ceil(torch.tensor(self.rc / cell_size)).item())

        offs = torch.tensor(
            [(dx, dy, dz)
             for dx in range(-R, R + 1)
             for dy in range(-R, R + 1)
             for dz in range(-R, R + 1)],
            dtype=torch.long,
        )
        self.register_buffer("offs", offs, persistent=False)  # (K,3)
        self.R = R
        self.cell_size = float(cell_size)

        # Debug throttle
        self._dbg = 0

    @torch.no_grad()
    def forward(
        self,
        coord: torch.Tensor,
        *,
        ind_1: torch.Tensor,
        cell: torch.Tensor,
        rc: Optional[float] = None,     # accepted for compatibility (ignored)
        **kwargs,                       # accepted for compatibility (ignored)
    ) -> Dict[str, torch.Tensor]:
        device = coord.device
        dtype = coord.dtype

        # Optional sanity check if caller provides rc
        if rc is not None:
            rc_f = float(rc)
            if abs(rc_f - self.rc) > 1e-6:
                raise ValueError(f"CellListNLBatched initialized with rc={self.rc} but called with rc={rc_f}")

        rc2 = self.rc * self.rc
        cell_size = self.cell_size

        # structure id per atom
        sid = ind_1[:, 0] if ind_1.ndim == 2 else ind_1
        sid = sid.to(dtype=torch.long, device=device)

        N = int(coord.shape[0])
        if N == 0:
            return {
                "ind_2": torch.zeros((0, 2), dtype=torch.long, device=device),
                "shift": torch.zeros((0, 3), dtype=torch.long, device=device),
            }

        B = int(sid.max().item()) + 1

        # --------- PBC cell lengths per structure (orthorhombic only) ----------
        if cell.ndim == 2:
            L = _get_box_lengths(cell.to(device=device, dtype=dtype))  # (3,)
            Lb = L[None, :].expand(B, 3)                               # (B,3)
        elif cell.ndim == 3:
            if cell.shape[0] != B:
                raise ValueError(f"Per-structure cell has shape {tuple(cell.shape)} but B={B}")
            cell_d = cell.to(device=device, dtype=dtype)
            offdiag = cell_d - torch.diag_embed(torch.diagonal(cell_d, dim1=1, dim2=2))
            if torch.max(torch.abs(offdiag)).item() > 1e-6:
                raise NotImplementedError(
                    "CellListNLBatched supports orthorhombic per-structure cells only (near-diagonal 3x3)."
                )
            Lb = torch.abs(torch.diagonal(cell_d, dim1=1, dim2=2))      # (B,3)
        else:
            raise ValueError(f"Unexpected cell shape {tuple(cell.shape)}")

        # --------- Bin atoms into cell-list grid (using cell_size, NOT rc) ----------
        ngrid = torch.floor(Lb / cell_size).to(torch.long).clamp_min(1)  # (B,3)

        # Wrap coords into [0, L)
        L_atom = Lb[sid]                              # (N,3)
        rel = torch.remainder(coord, L_atom)          # (N,3) in [0, L)
        cxyz = torch.floor(rel / cell_size).to(torch.long)
        cxyz = torch.minimum(cxyz, (ngrid[sid] - 1))

        # linearize within structure
        lin = cxyz[:, 0] + ngrid[sid, 0] * (cxyz[:, 1] + ngrid[sid, 1] * cxyz[:, 2])

        # global cell ids via offsets per structure
        ncell_struct = (ngrid[:, 0] * ngrid[:, 1] * ngrid[:, 2]).to(torch.long)  # (B,)
        cell_offsets = torch.zeros((B + 1,), device=device, dtype=torch.long)
        cell_offsets[1:] = torch.cumsum(ncell_struct, dim=0)
        lin_global = lin + cell_offsets[sid]

        # sort atoms by global cell id
        order_sorted = torch.argsort(lin_global)      # (N,)
        lin_sorted = lin_global[order_sorted]         # (N,)

        uniq_cells, counts = torch.unique_consecutive(lin_sorted, return_counts=True)  # (C,), (C,)
        C = int(uniq_cells.numel())
        if C == 0:
            return {
                "ind_2": torch.zeros((0, 2), dtype=torch.long, device=device),
                "shift": torch.zeros((0, 3), dtype=torch.long, device=device),
            }

        starts = torch.zeros((C + 1,), device=device, dtype=torch.long)
        starts[1:] = torch.cumsum(counts, dim=0)
        max_count = int(counts.max().item())

        if self._dbg < 5:
            self._dbg += 1
            K = int(self.offs.shape[0])
            print(
                "[NLBatched] N=", int(N),
                "C=", int(C),
                "max_count=", int(max_count),
                "mean_count=", float(counts.float().mean().item()),
                "p90_count=", float(torch.quantile(counts.float(), 0.9).item()),
                "cell_size=", float(cell_size),
                "R=", int(self.R),
                "K=", int(K),
            )

        # --------- Build padded atoms-per-cell table: cell_atoms (C, M) ----------
        # cell_atoms[c, t] = atom index (original indexing) of t-th atom in that cell
        cell_atoms = torch.full((C, max_count), -1, device=device, dtype=torch.long)

        # Vectorized fill without Python loops:
        # flat_sorted_idx = [starts[c]..starts[c+1]) concatenated; seg gives cell index for each
        lens = (starts[1:] - starts[:-1]).to(torch.long)
        total = int(lens.sum().item())
        if total > 0:
            seg = torch.repeat_interleave(torch.arange(C, device=device), lens)  # (total,)
            base = torch.arange(total, device=device, dtype=torch.long)
            seg_startpos = torch.cumsum(lens, dim=0) - lens
            off = base - seg_startpos[seg]
            flat_sorted_idx = starts[seg] + off

            pos_in_seg = flat_sorted_idx - starts[seg]             # (total,)
            atoms = order_sorted[flat_sorted_idx]                  # (total,)
            cell_atoms[seg, pos_in_seg] = atoms

        # --------- Neighbor cell mapping (stencil radius R, K=(2R+1)^3) ----------
        offs = self.offs.to(device=device)                         # (K,3)
        K = int(offs.shape[0])

        # cell_sid: structure id per occupied cell
        cell_sid = torch.searchsorted(cell_offsets[1:], uniq_cells, right=False)  # (C,)
        local_cell = uniq_cells - cell_offsets[cell_sid]                         # (C,)

        nx_s = ngrid[cell_sid, 0]
        ny_s = ngrid[cell_sid, 1]
        ix = local_cell % nx_s
        iy = (local_cell // nx_s) % ny_s
        iz = local_cell // (nx_s * ny_s)

        nix = ix[:, None] + offs[None, :, 0]    # (C,K)
        niy = iy[:, None] + offs[None, :, 1]
        niz = iz[:, None] + offs[None, :, 2]

        # Wrap (PBC) + record image shifts for general R (not just -1/0/1)
        nxv = ngrid[cell_sid, 0][:, None]       # (C,1)
        nyv = ngrid[cell_sid, 1][:, None]
        nzv = ngrid[cell_sid, 2][:, None]

        nix_wrapped = nix.remainder(nxv)
        niy_wrapped = niy.remainder(nyv)
        niz_wrapped = niz.remainder(nzv)

        sx = (nix - nix_wrapped) // nxv
        sy = (niy - niy_wrapped) // nyv
        sz = (niz - niz_wrapped) // nzv

        nix = nix_wrapped
        niy = niy_wrapped
        niz = niz_wrapped

        nshift = torch.stack([sx, sy, sz], dim=-1)  # (C,K,3)

        nlocal = nix + (ngrid[cell_sid, 0])[:, None] * (niy + (ngrid[cell_sid, 1])[:, None] * niz)
        ncell_global = nlocal + cell_offsets[cell_sid][:, None]  # (C,K)

        # map neighbor global cell ids -> indices in uniq_cells (or invalid)
        pos = torch.searchsorted(uniq_cells, ncell_global)        # (C,K)
        pos_safe = pos.clamp_max(C - 1)
        present = (pos < C) & (uniq_cells[pos_safe] == ncell_global)  # (C,K)
        neigh_cell_idx = pos_safe                                      # (C,K) valid where present

        # Precompute box lengths per occupied cell (for PBC shift->Cartesian)
        L_cell = Lb[cell_sid]  # (C,3)

        src_atoms = cell_atoms              # (C,M)
        src_valid = src_atoms >= 0

        ind_i_all = []
        ind_j_all = []
        shift_all = []
    

        # --------- Build edges per neighbor direction (constant K loop) ----------
        # Note: This is “vectorized within each neighbor offset” but still does dense MxM.
        # The binning change (cell_size < rc) is what prevents M from exploding.
        for k in range(K):
            ok = present[:, k]
            if not bool(ok.any().item()):
                continue

            tgt_cells = neigh_cell_idx[:, k]          # (C,)
            tgt_atoms = cell_atoms[tgt_cells]         # (C,M)
            tgt_valid = tgt_atoms >= 0

            # restrict to ok cells
            sa = src_atoms[ok]                        # (Ck,M)
            sv = src_valid[ok]
            ta = tgt_atoms[ok]                        # (Ck,M)
            tv = tgt_valid[ok]
            sh = nshift[ok, k]                        # (Ck,3)
            Lp = L_cell[ok]                           # (Ck,3)

            Msrc = sa.shape[1]
            Mtgt = ta.shape[1]

            ii = sa[:, :, None].expand(-1, Msrc, Mtgt)   # (Ck,Msrc,Mtgt)
            jj = ta[:, None, :].expand(-1, Msrc, Mtgt)   # (Ck,Msrc,Mtgt)

            mask = sv[:, :, None] & tv[:, None, :] & (ii != jj)
            if not mask.any():
                continue

            i = ii[mask]  # (E,)
            j = jj[mask]  # (E,)

            sh_exp = sh[:, None, None, :].expand(-1, Msrc, Mtgt, 3)  # (Ck,Msrc,Mtgt,3)
            shift_pairs = sh_exp[mask]                                # (E,3)

            # Convert integer shift -> Cartesian using orthorhombic lengths
            L_exp = Lp[:, None, None, :].expand(-1, Msrc, Mtgt, 3)     # (Ck,Msrc,Mtgt,3)
            L_pairs = L_exp[mask]                                       # (E,3)

            rij = coord[j] - coord[i]
            rij = rij + shift_pairs.to(dtype) * L_pairs.to(dtype)

            d2 = (rij * rij).sum(dim=1)
            keep = d2 <= rc2
            if keep.any():
                ind_i_all.append(i[keep])
                ind_j_all.append(j[keep])
                shift_all.append(shift_pairs[keep])


        if not ind_i_all:
            return {
                "ind_2": torch.zeros((0, 2), dtype=torch.long, device=device),
                "shift": torch.zeros((0, 3), dtype=torch.long, device=device),
            }

        ind_i = torch.cat(ind_i_all, dim=0)
        ind_j = torch.cat(ind_j_all, dim=0)
        shift = torch.cat(shift_all, dim=0).to(torch.long)

        ind_2 = torch.stack([ind_i, ind_j], dim=1).to(torch.long)
        return {"ind_2": ind_2, "shift": shift}