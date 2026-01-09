# -*- coding: utf-8 -*-
"""
PyTorch translation of PiNet2 (rank=1 or rank=3), non-weighted PIX/Dot only.

This mirrors the TensorFlow reference implementation documented at:
https://teoroo-cmc.github.io/PiNN/master/usage/pinet2/

Key pieces (rank=3 / P3, non-weighted):

- d3: unit bond directions on pairs
      d3 = diff / ||diff||                      # (n_pairs, 3)

- p1: per-atom invariant properties
      shape: (n_atoms, n_prop)

- p3: per-atom equivariant (rank-3) properties
      shape: (n_atoms, 3, n_chan)
      initialized as zeros at network entry

- InvarLayer produces two tensors:
    * p1_new : per-atom invariant update
              shape (n_atoms, n_prop')
    * i_pair : per-pair invariant interaction channels
              shape (n_pairs, n_i)

  In rank=3, i_pair is split into two branches:
    i_pair -> [i_pair_0, i_pair_3]
  where i_pair_3 gates the equivariant (p3) update.

- GCBlock (rank=3) logic mirrors TF PiNet2:
    1) Split per-pair interaction channels:
         i_pair -> [i_pair_0, i_pair_3]
    2) Use i_pair_3 to gate equivariant update:
         ix = PIX([ind_2, p3])
         ix = Scale(ix, i_pair_3)
         ix += Scale(d3[:, :, None], i_pair_3)
         (optional) ix += Scale(t3[:, :, None], i_pair_3)
    3) Scatter interactions back to atoms:
         p3 = IP([ind_2, p3, ix])
         p3 = PP_no_bias_no_act(p3)
    4) Form invariant–equivariant coupling:
         dotted_p3 = Dot(p3)                    # per-atom
         concat([p1_new, dotted_p3])
         -> pp_layer -> split -> [p1_next, p3_scale]
    5) Rescale equivariant features:
         p3_next = p3 * p3_scale[:, None, :]

- Output heads accumulate per-atom energies; pooling to structures is handled
  outside the network (PiNetPotentialTorch).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union, Callable, Dict, Any

import torch
import torch.nn as nn

from pinn.networks.pinet_torch import (
    PreprocessLayerTorch,
    CutoffFuncTorch,
    PolynomialBasisTorch,
    GaussianBasisTorch,
    ANNOutputTorch,
    OutLayerTorch,
    ResUpdateTorch,
    PILayerTorch,
    FFLayerTorch,
    IPLayerTorch,
)


def compute_torsion_boost_t3(
    *,
    ind_2: torch.Tensor,
    d3: torch.Tensor,
    n_atoms: int,
    fc_edge: torch.Tensor,
    eps: float = 1e-3,   # larger eps for float32 stability
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      t3:      (n_pairs,3) perpendicular direction, already damped near degeneracy
      tb_gate: (n_pairs,1) scalar gate (cutoff + degeneracy)
    """
    if ind_2.dtype != torch.long:
        ind_2 = ind_2.long()
    if d3.numel() == 0:
        return d3.new_zeros((0, 3)), d3.new_zeros((0, 1))

    i = ind_2[:, 0]
    fc = fc_edge[:, None] if fc_edge.ndim == 1 else fc_edge  # (n_pairs,1)

    # 1) v_i = sum_{i->k} fc(i,k) * d3(i,k)
    v = d3.new_zeros((int(n_atoms), 3))
    v.index_add_(0, i, d3 * fc)

    v_i = v[i]  # (n_pairs,3)

    # 2) w = v_i - (v_i·d3) d3   (perpendicular rejection)
    proj = torch.sum(v_i * d3, dim=-1, keepdim=True)          # (n_pairs,1)
    w = v_i - proj * d3                                       # (n_pairs,3)

    # 3) Robust norm and degeneracy gate
    w2 = torch.sum(w * w, dim=-1, keepdim=True)               # (n_pairs,1)

    # Reference scale tied to eps (tune factor if needed)
    w2_ref = (10.0 * eps) ** 2                                # (scalar)

    # Smooth rational gate: ~0 when w2<<w2_ref, ~1 when w2>>w2_ref
    g_deg = w2 / (w2 + w2_ref)

    # 4) Normalize, but also damp the direction by g_deg to kill stiff gradients near w=0
    inv = torch.rsqrt(w2 + eps * eps)
    t3 = w * inv * g_deg                                      # (n_pairs,3)

    # 5) Cutoff fade (keep non-negative and decaying to 0 at rc)
    g_cut = fc * fc                                           # (n_pairs,1)

    tb_gate = g_deg * g_cut                                   # (n_pairs,1)
    return t3, tb_gate

class ScaleLayerTorch(nn.Module):
    """Equivariant scaling: X'[.., x, r] = X[.., x, r] * s[.., r]."""

    def forward(self, px: torch.Tensor, p1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            px: (..., x, r) tensor
            p1: (..., r) tensor (same leading dim as px)
        Returns:
            scaled: (..., x, r)
        """
        return px * p1[:, None, :]


class PIXLayerTorch(nn.Module):
    """Non-weighted PIXLayer: broadcast atom tensor to pair tensor by taking j-side."""

    def forward(self, ind_2: torch.Tensor, px: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ind_2: (n_pairs, 2) long, pairs (i, j)
            px:    (n_atoms, x, r) equivariant property
        Returns:
            ix:    (n_pairs, x, r) = px[j]
        """
        if ind_2.dtype != torch.long:
            ind_2 = ind_2.long()
        j = ind_2[:, 1]
        return px[j]


class DotLayerTorch(nn.Module):
    """Non-weighted DotLayer: einsum('ixr,ixr->ir')."""

    def forward(self, px: torch.Tensor) -> torch.Tensor:
        """
        Args:
            px: (n_atoms, x, r)
        Returns:
            dotted: (n_atoms, r)
        """
        return torch.einsum("ixr,ixr->ir", px, px)


class IPLayerEqTorch(nn.Module):
    """Equivariant IP scatter-add over source atom index i (ind_2[:,0])."""

    def forward(self, ind_2: torch.Tensor, px: torch.Tensor, ix: torch.Tensor) -> torch.Tensor:
        """Equivariant IP scatter-add over receiver atom index i (ind_2[:, 0]).

        TF parity note:
            TF IPLayer returns a tensor whose last dimension follows the interaction
            tensor (ix), not the input atom tensor (px). This allows p3 to start with
            1 channel and expand through gating.

        Args:
            ind_2: (n_pairs, 2) long, (i, j)
            px:    (n_atoms, x, C_old) used only for n_atoms / device / dtype
            ix:    (n_pairs, x, C_new) interactions to scatter onto i

        Returns:
            out:   (n_atoms, x, C_new)
        """
        if ind_2.dtype != torch.long:
            ind_2 = ind_2.long()

        n_atoms = int(px.shape[0])
        x_dim = int(ix.shape[1])
        c_dim = int(ix.shape[2])

        out = torch.zeros((n_atoms, x_dim, c_dim), dtype=ix.dtype, device=ix.device)

        # Empty edge list is valid (e.g., tiny system or very small cutoff)
        if ix.numel() == 0:
            return out

        i = ind_2[:, 0]
        out.index_add_(0, i, ix)
        return out


class InvarLayerTorch(nn.Module):
    """Invariant layer in PiNet2: produces (p1_new, i_pair)."""
    def __init__(
        self,
        *,
        pp_nodes: Sequence[int],
        pi_nodes: Sequence[int],
        ii_nodes: Sequence[int],
        n_basis: int,
        activation: Optional[Union[str, Callable[[torch.Tensor], torch.Tensor]]] = "tanh",
    ) -> None:
        """Initialize invariant block.

        Args:
            pp_nodes: MLP widths for per-atom preprocessing and postprocessing of p1.
            pi_nodes: MLP widths for pair interaction features.
            ii_nodes: MLP widths for pair interaction mixing.
            n_basis:  number of radial basis functions.
            activation: activation function name/callable.
        """
        super().__init__()

        self.pp_pre = FFLayerTorch(pp_nodes, activation=activation, use_bias=True) if len(pp_nodes) else nn.Identity()
        self.pi = PILayerTorch(pi_nodes, n_basis=int(n_basis), activation=activation, use_bias=True)
        self.ii = FFLayerTorch(ii_nodes, activation=activation, use_bias=False)
        self.ip = IPLayerTorch()
        # TF InvarLayer uses a biasless PP at the end (activation is still applied)
        self.pp_post = FFLayerTorch(pp_nodes, activation=activation, use_bias=False) if len(pp_nodes) else nn.Identity()

    def forward(
        self, *, ind_2: torch.Tensor, p1: torch.Tensor, basis: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute invariant update.

        Args:
            ind_2: (n_pairs, 2)
            p1:    (n_atoms, n_prop)
            basis: (n_pairs, n_basis)

        Returns:
            p1_new: (n_atoms, n_prop')
            i_pair: (n_pairs, n_i) pair-wise invariant interaction channels
        """
        p1_in = self.pp_pre(p1) if not isinstance(self.pp_pre, nn.Identity) else p1

        # Pair-wise interaction channels
        i_pair = self.pi(ind_2, p1_in, basis)
        i_pair = self.ii(i_pair)

        # Updated atom features
        p1_new = self.ip(ind_2, p1_in, i_pair)
        p1_new = self.pp_post(p1_new) if not isinstance(self.pp_post, nn.Identity) else p1_new

        return p1_new, i_pair
    
class EquivarLayerTorch(nn.Module):
    """Equivariant (rank=3) layer in PiNet2: produces (p3, dotted_p3).

    Faithful TF behavior:
      ix = PIX([ind_2, p3])
      ix = Scale([ix, i1])
      ix += Scale([d3[:, :, None], i1])
      p3 = IP([ind_2, p3, ix])
      p3 = PP_no_bias_no_act(p3)
      dotted = Dot(p3)
    """

    def _lift_dir(self, direction: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Lift scalar pair channels into a vector message along a direction.

        Args:
            direction: (n_pairs, 3) unit direction vector (e.g., d3 or t3).
            gate:      (n_pairs, C) scalar channels to scale each direction component.

        Returns:
            (n_pairs, 3, C) vector message with per-channel scaling.
        """
        return direction[:, :, None] * gate[:, None, :]

    def __init__(
        self,
        *,
        pp_nodes: Sequence[int],
        torsion_boost: bool = False,
    ) -> None:
        super().__init__()
        self.torsion_boost = bool(torsion_boost)
        self.pix = PIXLayerTorch()
        self.scale = ScaleLayerTorch()
        self.ip = IPLayerEqTorch()

        # TF EquivarLayer uses activation=None and use_bias=False in its PP stage.
        self.pp = FFLayerTorch(pp_nodes, activation=None, use_bias=False) if len(pp_nodes) else nn.Identity()
        self.dot = DotLayerTorch()

        # --- Knob A: learnable per-pair torsion relevance gate (scalar) ---
        # We know i1 channel dim from pp_nodes used to build EquivarLayerTorch (ppx_nodes=[pp_nodes[-1]])
        C = int(pp_nodes[-1])  # i1 has shape (n_pairs, C)

        # Gate takes concat([i1, s]) where s is (n_pairs,1) -> input dim is C+1
        self.tb_pair_gate = nn.Linear(C + 1, 1, bias=True)

        #     Make i1 weights zero so the only initial driver is s.
        with torch.no_grad():
            nn.init.zeros_(self.tb_pair_gate.weight)      # (1, C+1)
            self.tb_pair_gate.bias.fill_(-1.0)            # not -2.0 (too dead with temp>1)
            self.tb_pair_gate.weight[0, -1] = 2.0         # positive weight on s channel

        # Temperature: don’t go too sharp initially
        self.tb_temp = 2.0   # try 2.0 first (then 3.0 if still not selective)

        # TB channel adapter: keep it, init near identity
        self.tb_proj = nn.Linear(C, C, bias=False)
        with torch.no_grad():
            nn.init.eye_(self.tb_proj.weight)

        # Debug toggle (prefer False by default; enable from tests/params)
        self.debug_tb = False

        # activation is kept in signature to stay API-compatible, but not used here (TF defines ii_layer but doesn't use it).

    def forward(
        self,
        *,
        ind_2: torch.Tensor,
        p3: torch.Tensor,
        i1: torch.Tensor,
        d3: torch.Tensor,
        t3: Optional[torch.Tensor] = None,
        fc_edge: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ind_2: (n_pairs, 2) long, (i, j)
            p3:    (n_atoms, 3, C_in)
            i1:    (n_pairs, C_in) pair-wise gate (rank-3 branch)
            d3:    (n_pairs, 3) unit bond directions
            t3:    (n_pairs, 3) TB direction (already damped if you use the new compute_torsion_boost_t3)
            fc_edge: (n_pairs, 1) TB scalar gate (tb_gate)
        """
        ix = self.pix(ind_2, p3)
        ix = self.scale(ix, i1)
        ix = ix + self._lift_dir(d3, i1)

        if self.torsion_boost:
            if t3 is None:
                raise ValueError("torsion_boost=True requires t3.")
            if fc_edge is None:
                raise ValueError("torsion_boost=True requires tb_gate (fc_edge).")
            # --- Knob A: learned torsion relevance gate ---
            # g_learn: (n_pairs, 1) in (0,1)
            s = (t3 * t3).sum(dim=-1, keepdim=True)          # (n_pairs,1) relevance magnitude
            s = s / (s.detach().mean() + 1e-6)               # dimensionless, mean ~ 1
            gate_in = torch.cat([i1, s], dim=-1)
            logit = self.tb_pair_gate(gate_in)          # (n_pairs,1)
            g_learn = torch.sigmoid(self.tb_temp * logit)    

            # fc_edge is tb_gate: (n_pairs, 1)
            tb_eff = fc_edge * g_learn

            # --- TB has its own channel mixing adapter W_tb ---

            i_tb = self.tb_proj(i1)   # (n_pairs, C)

            if self.debug_tb:
                if tb_eff.numel() == 0:
                    print("[TB] empty edge list (no pairs inside cutoff) — skipping TB stats")
                else:
                    # How alive is the geometric gate?
                    tb_eff_mean = float(tb_eff.mean().detach().cpu())
                    tb_eff_max  = float(tb_eff.max().detach().cpu())
                    live_frac   = float((tb_eff > 1e-6).float().mean().detach().cpu())

                    # How big is the TB message compared to the base directional message?
                    msg_d3 = self._lift_dir(d3, i1)                  # (n_pairs,3,C)
                    msg_tb = self._lift_dir(t3, i_tb * tb_eff)       # (n_pairs,3,C)

                    # robust RMS for non-empty tensors
                    d3_rms = float(msg_d3.pow(2).mean().sqrt().detach().cpu())
                    tb_rms = float(msg_tb.pow(2).mean().sqrt().detach().cpu())
                    ratio  = tb_rms / (d3_rms + 1e-12)

                    print(
                        f"[TB] tb_eff mean={tb_eff_mean:.3e} max={tb_eff_max:.3e} "
                        f"live={live_frac*100:.1f}% | msg_rms tb/d3={ratio:.3e}"
                    )

                    g_mean = float(g_learn.mean().detach().cpu())
                    g_max  = float(g_learn.max().detach().cpu())
                    g_min  = float(g_learn.min().detach().cpu())

                    # how close W_tb is to identity (if you initialized as identity)
                    w = self.tb_proj.weight
                    ident_err = float((w - torch.eye(w.shape[0], device=w.device, dtype=w.dtype)).pow(2).mean().sqrt().detach().cpu())

                    print(f"[TB] g_learn mean={g_mean:.3e} min={g_min:.3e} max={g_max:.3e} | Wtb_rmse_from_I={ident_err:.3e}")

            ix = ix + self._lift_dir(t3, i_tb * tb_eff)

        p3_new = self.ip(ind_2, p3, ix)
        p3_new = self.pp(p3_new) if not isinstance(self.pp, nn.Identity) else p3_new
        dotted = self.dot(p3_new)
        return p3_new, dotted


class GCBlock2Torch(nn.Module):
    """One PiNet2 GCBlock faithful to TF GCBlock.call() for rank=3 (non-weighted)."""

    def __init__(
        self,
        *,
        rank: int,
        pp_nodes: Sequence[int],
        pi_nodes: Sequence[int],
        ii_nodes: Sequence[int],
        n_basis: int,
        activation: str,
        torsion_boost: bool = False,
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.n_props = int(self.rank // 2) + 1  # rank=3 -> 2 branches

        if len(ii_nodes) == 0 or len(pp_nodes) == 0:
            raise ValueError("ii_nodes and pp_nodes must be non-empty for PiNet2 GCBlock")

        # TF (pinn/networks/pinet2.py GCBlock.__init__):
        #   ii1_nodes[-1] *= n_props
        #   pp1_nodes[-1]  = ii_nodes[-1] * n_props
        ii1_nodes = list(ii_nodes)
        pp1_nodes = list(pp_nodes)

        ii1_nodes[-1] = int(ii1_nodes[-1]) * int(self.n_props)
        pp1_nodes[-1] = (int(ii_nodes[-1])) * int(self.n_props)

        self.invar_p1_layer = InvarLayerTorch(
            pp_nodes=pp_nodes,
            pi_nodes=pi_nodes,
            ii_nodes=ii1_nodes,
            n_basis=n_basis,
            activation=activation,
        )

        self.pp_layer = FFLayerTorch(pp1_nodes, activation=activation, use_bias=True)

        if self.rank >= 3:
            ppx_nodes = [int(pp_nodes[-1])]
            self.equivar_p3_layer = EquivarLayerTorch(
                pp_nodes=ppx_nodes,
                torsion_boost=bool(torsion_boost),
            )
            self.scale3_layer = ScaleLayerTorch()

    def forward(self, tensors: Dict[str, torch.Tensor], basis: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute new tensors dict (p1 and p3) matching TF GCBlock output (rank=3)."""
        ind_2 = tensors["ind_2"]

        # 1) Invariant stage: per-atom update and per-pair interaction channels
        p1, i_pair = self.invar_p1_layer(ind_2=ind_2, p1=tensors["p1"], basis=basis)

        # 2) Split pair-wise interaction channels into branches (rank=3 -> 2 branches)
        i1s = torch.chunk(i_pair, chunks=self.n_props, dim=-1)

        px_list = [p1]

        # 3) Rank-3 equivariant stage gated by pair-wise branch i1s[1]
        if self.rank >= 3:
            p3, dotted_p3 = self.equivar_p3_layer(
                ind_2=ind_2,
                p3=tensors["p3"],
                i1=i1s[1],
                d3=tensors["d3"],
                t3=tensors.get("t3_unit", None),
                fc_edge=tensors.get("tb_gate", None),
            )
            px_list.append(dotted_p3)

            # --- Knob D: torsion invariant summary into scalar mixing ---
            # Build an axial vector per pair: a_pair = (d3 x t3) * tb_gate
            if self.equivar_p3_layer.torsion_boost:
                t3 = tensors.get("t3_unit", None)
                tb_gate = tensors.get("tb_gate", None)
                if t3 is not None and tb_gate is not None and t3.numel() != 0:
                    # axial per-pair vector (n_pairs,3)
                    a_pair = torch.cross(tensors["d3"], t3, dim=-1) * tb_gate  # tb_gate (n_pairs,1) broadcasts

                    # scatter to atoms i = ind_2[:,0] -> (n_atoms,3)
                    i = ind_2[:, 0]
                    n_atoms = int(tensors["p1"].shape[0])
                    a_atom = torch.zeros((n_atoms, 3), dtype=a_pair.dtype, device=a_pair.device)
                    a_atom.index_add_(0, i, a_pair)

                    # invariant torsion scalar per atom: |a_atom|^2 -> (n_atoms,1)
                    tau = (a_atom * a_atom).sum(dim=-1, keepdim=True)
                else:
                    # keep shapes consistent even if empty edge list
                    tau = p1.new_zeros((p1.shape[0], 1))

                px_list.append(tau)

        # 4) Second invariant mixing (per-atom): concat([p1, dotted_p3]) -> pp_layer -> split
        p1t1 = self.pp_layer(torch.cat(px_list, dim=-1))
        pxt1 = torch.chunk(p1t1, chunks=self.n_props, dim=-1)

        out: Dict[str, torch.Tensor] = {"p1": pxt1[0]}

        # 5) Scale p3 by per-atom pxt1[1]
        if self.rank >= 3:
            out["p3"] = self.scale3_layer(p3, pxt1[1])

        return out


class PiNet2Torch(nn.Module):
    """Full PiNet2 forward (dict-in, per-atom energy out) for Torch backend."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.params = dict(params)
        self.rank = int(self.params.get("rank", 3))
        if self.rank not in (1, 3):
            raise NotImplementedError("Torch PiNet2 currently supports rank=1 or rank=3 only (p3).")

        atom_types = self.params["atom_types"]
        rc = float(self.params["rc"])
        cutoff_type = self.params.get("cutoff_type", "f1")
        basis_type = self.params.get("basis_type", "polynomial")
        n_basis = int(self.params.get("n_basis", 5))
        gamma = float(self.params.get("gamma", 3.0))
        center = self.params.get("center", None)

        pp_nodes = self.params.get("pp_nodes", [8, 8])
        pi_nodes = self.params.get("pi_nodes", [8, 8])
        ii_nodes = self.params.get("ii_nodes", [8, 8])
        out_nodes = self.params.get("out_nodes", [8, 8])
        out_units = int(self.params.get("out_units", 1))
        out_pool = self.params.get("out_pool", False)
        act = self.params.get("act", "tanh")
        depth = int(self.params.get("depth", 3))

        self.depth = depth
        self.preprocess = PreprocessLayerTorch(atom_types, rc)
        self.cutoff = CutoffFuncTorch(rc, str(cutoff_type))

        if str(basis_type).lower() == "polynomial":
            self.basis_fn = PolynomialBasisTorch(n_basis)
        elif str(basis_type).lower() == "gaussian":
            self.basis_fn = GaussianBasisTorch(center=center, gamma=gamma, rc=rc, n_basis=n_basis)
        else:
            raise ValueError(f"Unknown basis_type={basis_type!r}")

        torsion_boost = bool(self.params.get("torsion_boost", False))

        self.gc_blocks = nn.ModuleList(
            [
                GCBlock2Torch(
                    rank=self.rank,
                    pp_nodes=pp_nodes,
                    pi_nodes=pi_nodes,
                    ii_nodes=ii_nodes,
                    n_basis=n_basis,
                    activation=str(act),
                    torsion_boost=torsion_boost,
                )
                for _ in range(self.depth)
            ]
        )
        self.torsion_boost = torsion_boost
        self.out_layers = nn.ModuleList(
            [OutLayerTorch(out_nodes, out_units, activation=str(act), use_bias=True) for _ in range(self.depth)]
        )
        self.res_update1 = nn.ModuleList([ResUpdateTorch() for _ in range(self.depth)])
        self.res_update3 = nn.ModuleList([ResUpdateTorch() for _ in range(self.depth)]) if self.rank >= 3 else None
        self.ann_output = ANNOutputTorch(out_pool)
        self.debug_tensors = bool(self.params.get("debug_tensors", False))
        self._last_tensors: Dict[str, torch.Tensor] = {}

        self.preprocess = PreprocessLayerTorch(atom_types, rc)

    def forward(self, tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            tensors: dict with keys: coord, elems, ind_1 (and optionally cell/ind_2/...)
        Returns:
            per-atom energies (N,) or (N,1)  (pooling handled by PiNetPotentialTorch)
        """
        # At the top of forward()
        if not tensors.get("_preprocessed", False):
            tensors = self.preprocess(tensors)

        # PiNetTorch preprocess uses "prop" for scalar features; PiNet2 docs call it "p1".
        # Keep both keys to avoid confusion.
        tensors["p1"] = tensors["prop"]

        # Init equivariants
        n_atoms = int(tensors["ind_1"].shape[0])

        # Unit bond directions d3 (TF: diff / ||diff||)
        diff = tensors["diff"]  # (n_pairs,3)
        dnorm = torch.linalg.norm(diff, dim=-1, keepdim=True).clamp_min(1e-6)
        tensors["d3"] = diff / dnorm

        # Cutoff + basis (compute fc early so TB can use it)
        fc = self.cutoff(tensors["dist"])

        if self.debug_tensors:
            with torch.no_grad():
                d = tensors["dist"]
                if d.numel():
                    mx = float(d.max().item())
                    alive = float((fc > 0).float().sum().item())
                    beyond = float(((d >= self.cutoff.rc) & (fc > 0)).float().sum().item())
                    print(f"[DEBUG] max dist={mx:.3f}, fc>0 edges={alive:.0f}, fc>0 beyond rc={beyond:.0f}")

        tensors["fc"] = fc
        basis = self.basis_fn(tensors["dist"], fc=fc)

        # Optional torsion_boost geometry: cutoff-weighted t3
        if self.torsion_boost and self.rank >= 3:
            t3, tb_gate = compute_torsion_boost_t3(
                ind_2=tensors["ind_2"],
                d3=tensors["d3"],
                n_atoms=n_atoms,
                fc_edge=fc,
            )
            tensors["t3_unit"] = t3
            tensors["tb_gate"] = tb_gate
        else:
            tensors.pop("t3", None)
            tensors.pop("tb_gate", None)

        if self.rank >= 3 and "p3" not in tensors:
            tensors["p3"] = torch.zeros(
                (n_atoms, 3, 1),
                dtype=tensors["coord"].dtype,
                device=tensors["coord"].device,
            )

        # Add a debug hook
        if self.debug_tensors:
            self._last_tensors = {k: v for k, v in tensors.items() if isinstance(v, torch.Tensor)}

        # Output accumulator
        output = torch.zeros(
            (n_atoms, self.out_layers[0].out_units),
            dtype=tensors["coord"].dtype,
            device=tensors["coord"].device,
        )

        for i in range(self.depth):
            new_tensors = self.gc_blocks[i](tensors, basis)

            output = self.out_layers[i](tensors["ind_1"], new_tensors["p1"], output)

            tensors["p1"] = self.res_update1[i](tensors["p1"], new_tensors["p1"])
            tensors["prop"] = tensors["p1"]

            if self.rank >= 3:
                assert self.res_update3 is not None
                tensors["p3"] = self.res_update3[i](tensors["p3"], new_tensors["p3"])

        output = self.ann_output(tensors["ind_1"], output)
        return output