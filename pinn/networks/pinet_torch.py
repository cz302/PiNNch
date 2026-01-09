# -*- coding: utf-8 -*-
"""
PyTorch translation of PiNet building blocks.

This file mirrors the TensorFlow implementation in pinn/networks/pinet.py,
keeping the same tensor semantics and shape conventions. The goal is structural
equivalence first (shapes/flow), then numerical parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Union
from pinn.io.torch.preprocess_fns import build_nl_celllist, compute_diff_dist, atomic_onehot

import torch
import torch.nn as nn
import numpy as np

def _get_activation(act: Optional[Union[str, Callable[[torch.Tensor], torch.Tensor]]]) -> nn.Module:
    """
    Return a torch.nn.Module activation matching PiNet's string activations.

    Args:
        act: Activation name (e.g. "tanh", "relu") or a callable. If None, identity.

    Returns:
        A torch.nn.Module that applies the activation.
    """
    if act is None:
        return nn.Identity()
    if callable(act):
        # Wrap callable for Module compatibility.
        class _CallableAct(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
                return act(x)

        return _CallableAct()

    key = str(act).lower()
    if key == "tanh":
        return nn.Tanh()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key in ("silu", "swish"):
        return nn.SiLU()
    if key == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation: {act!r}")


class FFLayerTorch(nn.Module):
    """
    Feed-forward MLP block mirroring the TF FFLayer.

    TensorFlow Dense can infer input feature size lazily; in PyTorch we mimic
    that behavior using LazyLinear for the first layer.

    The module applies a stack of Linear(+activation) layers.

    Notes:
        - If use_bias=False, biases are disabled (used for IILayer behavior).
        - Output shape: (..., n_nodes[-1]).
    """

    def __init__(
        self,
        n_nodes: Sequence[int] = (64, 64),
        *,
        activation: Optional[Union[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if len(n_nodes) == 0:
            raise ValueError("n_nodes must have at least one element.")
        self.n_nodes = list(map(int, n_nodes))
        self.use_bias = bool(use_bias)

        act_mod = _get_activation(activation)
        layers: List[nn.Module] = []

        # First layer is lazy on input features (TF Dense style).
        layers.append(nn.LazyLinear(self.n_nodes[0], bias=self.use_bias))
        if not isinstance(act_mod, nn.Identity):
            layers.append(act_mod)

        # Subsequent layers have known in_features.
        for i in range(1, len(self.n_nodes)):
            layers.append(nn.Linear(self.n_nodes[i - 1], self.n_nodes[i], bias=self.use_bias))
            if not isinstance(act_mod, nn.Identity):
                layers.append(_get_activation(activation))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the feed-forward stack.

        Args:
            x: Input tensor of shape (..., n_in).

        Returns:
            Output tensor of shape (..., n_nodes[-1]).
        """
        return self.net(x)


class PILayerTorch(nn.Module):
    """
    Pair-to-interaction layer mirroring TF PILayer.

    Given:
      - ind_2: (n_pairs, 2) indices (i, j)
      - prop:  (n_atoms, n_prop) atomic properties
      - basis: (n_pairs, n_basis) distance basis values

    Produces:
      - inter: (n_pairs, d) where d = n_nodes[-1]

    Implementation detail (matches TF code):
      - FFLayer outputs (n_pairs, d * n_basis)
      - reshape to (n_pairs, d, n_basis)
      - contract over basis index: einsum("pcb,pb->pc")
    """

    def __init__(
        self,
        n_nodes: Sequence[int] = (64,),
        *,
        n_basis: int,
        activation: Optional[Union[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if len(n_nodes) == 0:
            raise ValueError("n_nodes must have at least one element.")
        self.n_nodes = list(map(int, n_nodes))
        self.n_basis = int(n_basis)

        n_nodes_iter = self.n_nodes.copy()
        n_nodes_iter[-1] = n_nodes_iter[-1] * self.n_basis  # output is flattened (d * b)

        self.ff_layer = FFLayerTorch(
            n_nodes_iter,
            activation=activation,
            use_bias=use_bias,
        )

    def forward(self, ind_2: torch.Tensor, prop: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise interaction channels.

        Args:
            ind_2: Long tensor (n_pairs, 2) with atom indices (i, j).
            prop:  Float tensor (n_atoms, n_prop).
            basis: Float tensor (n_pairs, n_basis).

        Returns:
            inter: Float tensor (n_pairs, d) where d = n_nodes[-1].
        """
        if ind_2.dtype != torch.long:
            ind_2 = ind_2.long()

        ind_i = ind_2[:, 0]
        ind_j = ind_2[:, 1]
        prop_i = prop[ind_i]
        prop_j = prop[ind_j]

        inter = torch.cat([prop_i, prop_j], dim=-1)  # (n_pairs, 2*n_prop)
        inter = self.ff_layer(inter)                 # (n_pairs, d*n_basis)

        d = self.n_nodes[-1]
        b = self.n_basis
        inter = inter.reshape(-1, d, b)              # (n_pairs, d, n_basis)
        inter = torch.einsum("pcb,pb->pc", inter, basis)  # (n_pairs, d)
        return inter


class IPLayerTorch(nn.Module):
    """
    Interaction-to-property layer mirroring TF IPLayer.

    TF code:
        return tf.math.unsorted_segment_sum(inter, ind_2[:, 0], n_atoms)

    PyTorch equivalent:
        out = zeros(n_atoms, n_inter)
        out.index_add_(0, ind_i, inter)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, ind_2: torch.Tensor, prop: torch.Tensor, inter: torch.Tensor) -> torch.Tensor:
        """
        Sum pair interactions onto source atoms (i = ind_2[:, 0]).

        Args:
            ind_2: Long tensor (n_pairs, 2)
            prop:  Tensor (n_atoms, n_prop) used only to infer n_atoms/device/dtype
            inter: Tensor (n_pairs, n_inter)

        Returns:
            new_prop: Tensor (n_atoms, n_inter)
        """
        if ind_2.dtype != torch.long:
            ind_2 = ind_2.long()

        n_atoms = prop.shape[0]
        ind_i = ind_2[:, 0]

        out = torch.zeros(
            (n_atoms, inter.shape[-1]),
            dtype=inter.dtype,
            device=inter.device,
        )
        out.index_add_(0, ind_i, inter)
        return out


class OutLayerTorch(nn.Module):
    """
    Output update layer mirroring TF OutLayer.

    TF:
        prop' = FFLayer(prop)
        output = Dense(out_units, use_bias=False)(prop') + prev_output
    """

    def __init__(
        self,
        n_nodes: Sequence[int],
        out_units: int,
        *,
        activation: Optional[Union[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.out_units = int(out_units)  
        self.ff_layer = FFLayerTorch(n_nodes, activation=activation, use_bias=use_bias)
        self.out_linear = nn.LazyLinear(int(out_units), bias=False)

    def forward(self, ind_1: torch.Tensor, prop: torch.Tensor, prev_output: torch.Tensor) -> torch.Tensor:
        """
        Update per-atom output.

        Args:
            ind_1: Unused here (kept for interface parity with TF).
            prop: (n_atoms, n_prop)
            prev_output: (n_atoms, out_units)

        Returns:
            output: (n_atoms, out_units)
        """
        x = self.ff_layer(prop)
        return self.out_linear(x) + prev_output


class ResUpdateTorch(nn.Module):
    """
    ResNet-like update mirroring TF ResUpdate.

    If old and new have same last-dim: return old + new
    else: return Linear(old)->new_dim (biasless) + new

    The linear is created lazily on first forward if needed.
    """

    def __init__(self) -> None:
        super().__init__()
        self._transform: Optional[nn.Module] = None
        self._in_dim: Optional[int] = None
        self._out_dim: Optional[int] = None

    def forward(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        """
        Args:
            old: (..., d_old)
            new: (..., d_new)

        Returns:
            updated: (..., d_new)
        """
        d_old = int(old.shape[-1])
        d_new = int(new.shape[-1])

        if d_old == d_new:
            return old + new

        # Lazily create and register the transform module.
        if self._transform is None or self._in_dim != d_old or self._out_dim != d_new:
            self._transform = nn.Linear(d_old, d_new, bias=False).to(device=old.device, dtype=old.dtype)
            self._in_dim = d_old
            self._out_dim = d_new
            # ensure it's registered as a submodule
            self.add_module("transform", self._transform)

        assert self._transform is not None
        return self._transform(old) + new


class GCBlockTorch(nn.Module):
    """
    Graph-convolution block mirroring the TensorFlow GCBlock in PiNet.

    Dataflow:
      1) prop2      = PPLayer(prop)                 # optional per-atom preprocessing
      2) inter      = PILayer(ind_2, prop2, basis)  # pair interactions using distance basis expansion
      3) inter      = IILayer(inter)                # biasless MLP on pair channels
      4) prop_block = IPLayer(ind_2, prop2, inter)  # scatter-add onto atoms (source index ind_2[:, 0])

    Note:
        Residual mixing (ResUpdate) is handled outside this block in the PiNet
        depth loop, matching the TensorFlow implementation.

    Returns:
        prop_block: Float tensor of shape (n_atoms, ii_nodes[-1]).
    """

    def __init__(
        self,
        pp_nodes: Sequence[int],
        pi_nodes: Sequence[int],
        ii_nodes: Sequence[int],
        *,
        n_basis: int,
        activation: Optional[Union[str, Callable[[torch.Tensor], torch.Tensor]]] = "tanh",
    ) -> None:
        """
        Args:
            pp_nodes: Hidden/output sizes for the per-atom preprocessing MLP (PPLayer).
            pi_nodes: Hidden/output sizes for the pair-interaction MLP (PILayer). The last
                element is d (interaction channels before basis contraction).
            ii_nodes: Hidden/output sizes for the interaction-interaction MLP (IILayer).
                The last element ii_nodes[-1] determines the per-atom block output size.
            n_basis: Number of distance basis functions (b).
            activation: Activation function name or callable.
        """
        super().__init__()
        # IILayer is FFLayer with use_bias=False
        self.pp_layer = (
            FFLayerTorch(pp_nodes, activation=activation, use_bias=True)
            if len(pp_nodes)
            else nn.Identity()
        )
        self.pi_layer = PILayerTorch(pi_nodes, n_basis=n_basis, activation=activation, use_bias=True)
        self.ii_layer = FFLayerTorch(ii_nodes, activation=activation, use_bias=False)
        self.ip_layer = IPLayerTorch()

    def forward(self, ind_2: torch.Tensor, prop: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        """
        Run one graph-convolution block (no residual update).

        Args:
            ind_2: Long tensor of shape (n_pairs, 2) containing (i, j) indices per pair.
            prop:  Float tensor of shape (n_atoms, d_in) containing per-atom features.
            basis: Float tensor of shape (n_pairs, n_basis) containing per-pair basis values.

        Returns:
            prop_block: Float tensor of shape (n_atoms, ii_nodes[-1]) containing the block output.
        """
        prop2 = self.pp_layer(prop) if not isinstance(self.pp_layer, nn.Identity) else prop
        inter = self.pi_layer(ind_2, prop2, basis)
        inter = self.ii_layer(inter)
        prop_block = self.ip_layer(ind_2, prop2, inter)
        return prop_block
    
class PiNetTorchCore(nn.Module):
    """
    Core PiNet interaction stack in PyTorch.

    This module mirrors the *depth loop* structure of the TensorFlow PiNet:
      - for each depth step:
          prop_i   = GCBlock(...)
          output   = OutLayer(prop_i, output)
          prop     = ResUpdate(prop, prop_i)

    It intentionally assumes preprocessing (neighbor list, distances, cutoff,
    basis construction, initial embeddings) is done upstream.

    Returns per-atom outputs (pre-pooling).
    """

    def __init__(
        self,
        *,
        depth: int,
        pp_nodes: Sequence[int],
        pi_nodes: Sequence[int],
        ii_nodes: Sequence[int],
        out_nodes: Sequence[int],
        out_units: int,
        n_basis: int,
        activation: Optional[Union[str, Callable[[torch.Tensor], torch.Tensor]]] = "tanh",
    ) -> None:
        """
        Args:
            depth: Number of GCBlocks / interaction stages.
            pp_nodes, pi_nodes, ii_nodes: Node lists passed to GCBlockTorch.
            out_nodes: Node list for OutLayerTorch internal FFLayerTorch.
            out_units: Output channels accumulated at each depth.
            n_basis: Number of basis functions used by PILayerTorch.
            activation: Activation function name or callable.
        """
        super().__init__()
        self.depth = int(depth)

        self.gc_blocks = nn.ModuleList(
            [
                GCBlockTorch(
                    pp_nodes=pp_nodes,
                    pi_nodes=pi_nodes,
                    ii_nodes=ii_nodes,
                    n_basis=n_basis,
                    activation=activation,
                )
                for _ in range(self.depth)
            ]
        )
        self.out_layers = nn.ModuleList(
            [
                OutLayerTorch(
                    n_nodes=out_nodes,
                    out_units=out_units,
                    activation=activation,
                    use_bias=True,
                )
                for _ in range(self.depth)
            ]
        )
        self.res_updates = nn.ModuleList([ResUpdateTorch() for _ in range(self.depth)])
        self.out_units = int(out_units)

    def forward(
        self,
        *,
        ind_1: torch.Tensor,
        ind_2: torch.Tensor,
        prop: torch.Tensor,
        basis: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the PiNet interaction stack (no final pooling).

        Args:
            ind_1: Long tensor used for pooling downstream (kept for interface parity).
            ind_2: Long tensor of shape (n_pairs, 2) pair indices (i, j).
            prop:  Float tensor of shape (n_atoms, d0) initial per-atom features.
            basis: Float tensor of shape (n_pairs, n_basis) per-pair basis values.

        Returns:
            output: Float tensor of shape (n_atoms, out_units) per-atom outputs (pre-pooling).
        """
        # Output accumulator (per atom).
        output = torch.zeros((prop.shape[0], self.out_units), dtype=prop.dtype, device=prop.device)

        tensors_prop = prop
        for i in range(self.depth):
            prop_i = self.gc_blocks[i](ind_2, tensors_prop, basis)
            output = self.out_layers[i](ind_1, prop_i, output)
            tensors_prop = self.res_updates[i](tensors_prop, prop_i)

        return output

class AtomicOnehotTorch(nn.Module):
    """
    One-hot embedding for atomic numbers.

    Mirrors pinn.layers.misc.AtomicOnehot:
      prop[i, a] = 1 if elems[i] == atom_types[a] else 0
    """

    def __init__(self, atom_types: Sequence[int] = (1, 6, 7, 8)) -> None:
        super().__init__()
        self.register_buffer("atom_types", torch.tensor(list(atom_types), dtype=torch.long))

    def forward(self, elems: torch.Tensor) -> torch.Tensor:
        """
        Args:
            elems: Long tensor of shape (n_atoms,) with atomic numbers.

        Returns:
            Boolean tensor of shape (n_atoms, n_types).
        """
        if elems.dtype != torch.long:
            elems = elems.long()
        return elems[:, None].eq(self.atom_types[None, :])


class CutoffFuncTorch(nn.Module):
    def __init__(self, rc: float = 5.0, cutoff_type: str = "f1") -> None:
        super().__init__()
        self.rc = float(rc)
        self.cutoff_type = str(cutoff_type).lower()

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Smooth cutoff that is guaranteed to satisfy:
          - fc(dist) = 0 for dist >= rc
          - fc(dist) >= 0 for all dist
        """
        rc = self.rc

        # Clamp normalized distance into [0, 1] so f1 cannot "wiggle" for dist>rc.
        x = (dist / rc).clamp(min=0.0, max=1.0)

        if self.cutoff_type == "f1":
            fc = 0.5 * (torch.cos(torch.pi * x) + 1.0)   # in [0,1], exactly 0 at x=1
        elif self.cutoff_type == "f2":
            one = dist.new_tensor(1.0)
            fc = (torch.tanh(1.0 - x) / torch.tanh(one)) ** 3
        else:  # hip
            fc = torch.cos(torch.pi * x / 2.0) ** 2

        # Explicitly zero out any self edges / numerical junk at <=0 distance.
        fc = torch.where(dist > 0.0, fc, dist.new_zeros(()).expand_as(fc))
        return fc


class PolynomialBasisTorch(nn.Module):
    """
    Polynomial basis layer.

    Mirrors pinn.layers.basis.PolynomialBasis:
      basis[:, b] = fc ** n_b
    where n_b is [1,2,...,n_basis] by default (same as TF code).
    """

    def __init__(self, n_basis: Union[int, Sequence[int]]) -> None:
        super().__init__()
        if isinstance(n_basis, int):
            orders = [i + 1 for i in range(int(n_basis))]
        else:
            orders = list(map(int, n_basis))
        self.register_buffer("orders", torch.tensor(orders, dtype=torch.long))

    def forward(self, dist: torch.Tensor, *, fc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist: Unused (kept for parity).
            fc: Cutoff values, shape (n_pairs,).

        Returns:
            basis: Tensor of shape (n_pairs, n_basis).
        """
        # Stack fc**order for each order in self.orders
        basis = torch.stack([fc ** int(o) for o in self.orders.tolist()], dim=1)
        return basis


class GaussianBasisTorch(nn.Module):
    """
    Gaussian basis layer mirroring pinn.layers.basis.GaussianBasis.
    """

    def __init__(
        self,
        *,
        center: Optional[Sequence[float]] = None,
        gamma: Optional[Union[float, Sequence[float]]] = None,
        rc: Optional[float] = None,
        n_basis: Optional[int] = None,
    ) -> None:
        super().__init__()
        if center is None:
            if rc is None or n_basis is None:
                raise ValueError("If center is None, both rc and n_basis must be provided.")
            center_arr = torch.linspace(0.0, float(rc), int(n_basis))
        else:
            center_arr = torch.tensor(list(center), dtype=torch.float32)

        if gamma is None:
            raise ValueError("gamma must be provided for GaussianBasisTorch.")
        if isinstance(gamma, (int, float)):
            gamma_arr = torch.full_like(center_arr, float(gamma))
        else:
            gamma_arr = torch.tensor(list(gamma), dtype=torch.float32)
            if gamma_arr.numel() != center_arr.numel():
                raise ValueError("gamma must have the same length as center.")

        self.register_buffer("center", center_arr)
        self.register_buffer("gamma", gamma_arr)

    def forward(self, dist: torch.Tensor, *, fc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            dist: Tensor of shape (n_pairs,)
            fc: Optional cutoff values (n_pairs,)

        Returns:
            basis: Tensor of shape (n_pairs, n_basis)
        """
        dist_ = dist[:, None]
        basis = torch.exp(-self.gamma[None, :] * (dist_ - self.center[None, :]) ** 2)
        if fc is not None:
            basis = basis * fc[:, None]
        return basis


class ANNOutputTorch(nn.Module):
    """
    Output pooling layer mirroring pinn.layers.misc.ANNOutput.

    If out_pool is falsy: returns per-atom outputs (squeezed to (n_atoms,) if out_units==1).
    If out_pool in {'sum','max','min','avg'}: reduces over atoms per structure using ind_1[:,0].
    """

    def __init__(self, out_pool: Union[bool, str]) -> None:
        super().__init__()
        self.out_pool = out_pool

    def forward(self, ind_1: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ind_1: Long tensor (n_atoms, 1) or (n_atoms, 2); uses ind_1[:,0] as structure id.
            output: Tensor (n_atoms, out_units)

        Returns:
            Tensor of shape:
              - (n_atoms, out_units) if no pooling
              - (n_structures, out_units) if pooled
            Then squeezed along last dim if out_units==1 (matching TF).
        """
        if ind_1.dtype != torch.long:
            ind_1 = ind_1.long()
        batch = ind_1[:, 0] if ind_1.ndim == 2 else ind_1
        if self.out_pool:
            n_struct = int(batch.max().item()) + 1 if batch.numel() else 0
            if self.out_pool == "sum":
                out = torch.zeros((n_struct, output.shape[1]), dtype=output.dtype, device=output.device)
                out.index_add_(0, batch, output)
            elif self.out_pool == "avg":
                out = torch.zeros((n_struct, output.shape[1]), dtype=output.dtype, device=output.device)
                out.index_add_(0, batch, output)
                cnt = torch.zeros((n_struct,), dtype=output.dtype, device=output.device)
                cnt.index_add_(0, batch, torch.ones_like(batch, dtype=output.dtype))
                out = out / cnt.clamp_min(1.0)[:, None]
            elif self.out_pool in ("max", "min"):
                # Simple (not super fast) scatter reduce for small test sizes.
                out = torch.full((n_struct, output.shape[1]),
                                 float("-inf") if self.out_pool == "max" else float("inf"),
                                 dtype=output.dtype, device=output.device)
                for i in range(output.shape[0]):
                    b = int(batch[i].item())
                    if self.out_pool == "max":
                        out[b] = torch.maximum(out[b], output[i])
                    else:
                        out[b] = torch.minimum(out[b], output[i])
            else:
                raise ValueError(f"Unknown out_pool={self.out_pool!r}")
            output = out

        # TF does tf.squeeze(output, axis=1) assuming out_units==1. We'll match that behavior:
        if output.ndim == 2 and output.shape[1] == 1:
            output = output[:, 0]
        return output


class NLRouter(nn.Module):
    def __init__(self, *, nl_pbc: nn.Module, nl_free: nn.Module, rc: float):
        super().__init__()
        self.rc = float(rc)
        self.nl_pbc = nl_pbc         # your CellListNLBatched (PBC fast path)
        self.nl_free = nl_free       # your CellListNLPyTorch (non-PBC + safe PBC fallback)

    @torch.no_grad()
    def forward(
        self,
        coord: torch.Tensor,
        *,
        ind_1: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        rc: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        rc_eff = self.rc if rc is None else float(rc)

        # ---- Non-PBC: keep existing path ----
        if cell is None:
            return build_nl_celllist(
                ind_1=ind_1,
                coord=coord,
                cell=None,
                rc=rc_eff,
                nl_builder=self.nl_free,
            )

        # ---- PBC: choose between batched (fast) vs CellListNLPyTorch (ASE-consistent fallback) ----
        # 1) Compute per-structure min box length (only what we need for mic_ok)
        #    We’ll be conservative: if anything looks non-orthorhombic, fall back.
        try:
            if cell.ndim == 2:
                H = cell
                offdiag = H - torch.diag(torch.diagonal(H))
                if torch.max(torch.abs(offdiag)).item() > 1e-6:
                    # triclinic-ish: batched builder doesn't support it; fallback
                    return build_nl_celllist(ind_1=ind_1, coord=coord, cell=cell, rc=rc_eff, nl_builder=self.nl_free)
                Lmin = float(torch.min(torch.abs(torch.diagonal(H))).item())
                mic_ok = (rc_eff <= 0.5 * Lmin)
            elif cell.ndim == 3:
                # per-structure cells
                H = cell
                offdiag = H - torch.diag_embed(torch.diagonal(H, dim1=1, dim2=2))
                if torch.max(torch.abs(offdiag)).item() > 1e-6:
                    return build_nl_celllist(ind_1=ind_1, coord=coord, cell=cell, rc=rc_eff, nl_builder=self.nl_free)
                L = torch.abs(torch.diagonal(H, dim1=1, dim2=2))  # (B,3)
                Lmin = float(torch.min(L).item())
                mic_ok = (rc_eff <= 0.5 * Lmin)
            else:
                # unexpected shape: fallback to robust path
                return build_nl_celllist(ind_1=ind_1, coord=coord, cell=cell, rc=rc_eff, nl_builder=self.nl_free)

        except Exception:
            # If anything goes sideways, prefer correctness
            return build_nl_celllist(ind_1=ind_1, coord=coord, cell=cell, rc=rc_eff, nl_builder=self.nl_free)

        # 2) If MIC is safe, use batched (fast)
        if mic_ok:
            return self.nl_pbc(coord, ind_1=ind_1, cell=cell)

        # 3) If MIC is NOT safe, fall back to CellListNLPyTorch, which already enumerates images ASE-style
        return build_nl_celllist(
            ind_1=ind_1,
            coord=coord,
            cell=cell,
            rc=rc_eff,
            nl_builder=self.nl_free,
        )

class PreprocessLayerTorch(nn.Module):
    """
    Preprocessing layer to build neighbor list + initial features.

    This mirrors the TF PreprocessLayer behavior at the interface level:
      - builds ind_2, dist, diff if missing
      - builds prop as one-hot embedding from elems if missing

    PBC behavior:
      - If tensors contains "cell", a minimum-image-convention (MIC) neighbor list
        is built per structure id (ind_1[:,0]).
      - If "cell" is absent, a free-space neighbor list is built.

    Notes:
        - By default, this layer uses CellListNLPyTorch (linked-cell) to build
          neighbors per structure id.
        - Brute-force helpers (_build_nl_free/_build_nl_mic) are kept as
          fallbacks/debug tools but are not used in forward() right now.
        - Cell convention matches PiNN TF nl._wrap_coord(): frac = coord @ inv(cell),
          coord = frac @ cell.
    """

    
    def __init__(self, atom_types: Sequence[int], rc: float) -> None:
        super().__init__()
        self.rc = float(rc)
        self.embed = AtomicOnehotTorch(atom_types)

        # PBC batched/vectorized NL
        from pinn.torch.celllist_batched import CellListNLBatched, BatchedCellListConfig
        self.nl_pbc = CellListNLBatched(BatchedCellListConfig(rc=rc))

        # Non-PBC per-structure NL 
        self.nl_free = CellListNLPyTorch(rc=rc)

        # IMPORTANT: runtime expects preprocess.nl to exist
        self.nl = NLRouter(nl_pbc=self.nl_pbc, nl_free=self.nl_free, rc=rc)
    

    @torch.no_grad()
    def _build_nl_free(self, ind_1: torch.Tensor, coord: torch.Tensor) -> dict:
        """
        Build a directed neighbor list i->j for atoms within cutoff (no PBC),
        grouped by structure id (ind_1[:,0]).

        Returns:
            dict with keys: ind_2 (long, (n_pairs,2)), dist ((n_pairs,)), diff ((n_pairs,3))
        """
        if ind_1.dtype != torch.long:
            ind_1 = ind_1.long()
        batch = ind_1[:, 0] if ind_1.ndim == 2 else ind_1

        ind2_list: List[torch.Tensor] = []
        dist_list: List[torch.Tensor] = []
        diff_list: List[torch.Tensor] = []

        for b in batch.unique(sorted=True).tolist():
            idx = (batch == b).nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            pos = coord[idx]  # (n,3)

            dmat = torch.cdist(pos, pos)  # (n,n)
            mask = (dmat < self.rc) & (dmat > 0.0)
            ii, jj = mask.nonzero(as_tuple=True)
            if ii.numel() == 0:
                continue

            gi = idx[ii]
            gj = idx[jj]
            diff = coord[gj] - coord[gi]
            dist = torch.linalg.norm(diff, dim=1)

            ind2_list.append(torch.stack([gi, gj], dim=1))
            dist_list.append(dist)
            diff_list.append(diff)

        if len(ind2_list) == 0:
            ind_2 = coord.new_zeros((0, 2), dtype=torch.long)
            dist = coord.new_zeros((0,), dtype=coord.dtype)
            diff = coord.new_zeros((0, 3), dtype=coord.dtype)
        else:
            ind_2 = torch.cat(ind2_list, dim=0)
            dist = torch.cat(dist_list, dim=0)
            diff = torch.cat(diff_list, dim=0)

        return {"ind_2": ind_2, "dist": dist, "diff": diff}

    @torch.no_grad()
    def _build_nl_mic(self, ind_1: torch.Tensor, coord: torch.Tensor, cell: torch.Tensor) -> dict:
        """
        Build a directed neighbor list i->j for atoms within cutoff using MIC (PBC),
        grouped by structure id (ind_1[:,0]).

        Args:
            ind_1: (n_atoms,1) or (n_atoms,2) long tensor; uses ind_1[:,0] as structure id.
            coord: (n_atoms,3) float tensor (Cartesian coordinates).
            cell:  either (3,3) or (n_struct,3,3) float tensor.

        Returns:
            dict with keys: ind_2 (long, (n_pairs,2)), dist ((n_pairs,)), diff ((n_pairs,3))
        """
        if ind_1.dtype != torch.long:
            ind_1 = ind_1.long()
        batch = ind_1[:, 0] if ind_1.ndim == 2 else ind_1

        # Determine whether cell is per-structure or global.
        cell_is_per_struct = cell.ndim == 3

        ind2_list: List[torch.Tensor] = []
        dist_list: List[torch.Tensor] = []
        diff_list: List[torch.Tensor] = []

        uniq = batch.unique(sorted=True)
        for b in uniq.tolist():
            idx = (batch == b).nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue

            H = cell[b] if cell_is_per_struct else cell  # (3,3)
            # Use float dtype/device consistent with coord
            H = H.to(device=coord.device, dtype=coord.dtype)
            H_inv = torch.linalg.inv(H)

            pos = coord[idx]                 # (n,3)
            frac = pos @ H_inv               # (n,3)  frac = coord @ inv(cell)
            # Pairwise fractional displacements: (n,n,3)
            dfrac = frac[None, :, :] - frac[:, None, :]
            dfrac = dfrac - torch.round(dfrac)  # MIC wrap into [-0.5, 0.5)

            # Back to Cartesian displacements (n,n,3)
            dcart = dfrac @ H
            # Distances (n,n)
            dmat = torch.linalg.norm(dcart, dim=2)

            mask = (dmat < self.rc) & (dmat > 0.0)
            ii, jj = mask.nonzero(as_tuple=True)
            if ii.numel() == 0:
                continue

            gi = idx[ii]
            gj = idx[jj]
            diff = dcart[ii, jj, :]  # already MIC displacement from i->j
            dist = dmat[ii, jj]

            ind2_list.append(torch.stack([gi, gj], dim=1))
            dist_list.append(dist)
            diff_list.append(diff)

        if len(ind2_list) == 0:
            ind_2 = coord.new_zeros((0, 2), dtype=torch.long)
            dist = coord.new_zeros((0,), dtype=coord.dtype)
            diff = coord.new_zeros((0, 3), dtype=coord.dtype)
        else:
            ind_2 = torch.cat(ind2_list, dim=0)
            dist = torch.cat(dist_list, dim=0)
            diff = torch.cat(diff_list, dim=0)

        return {"ind_2": ind_2, "dist": dist, "diff": diff}
    
    def forward(self, tensors: dict) -> dict:
        out = dict(tensors)

        # --- atom one-hot ---
        if "prop" not in out:
            prop = self.embed(out["elems"])
            out["prop"] = prop.to(dtype=out["coord"].dtype)

        # --- neighbor list (and shift for PBC) ---
        if "ind_2" not in out:
            cell = out.get("cell", None)
            nl = self.nl(out["coord"], ind_1=out["ind_1"], cell=cell)
            out.update(nl)  # keep extra keys (e.g., shift), BUT see below

            # CRITICAL: Never keep diff/dist produced by an NL builder (often no_grad),
            # because forces require diff/dist to be computed from coord with autograd.
            out.pop("diff", None)
            out.pop("dist", None)

        # --- edge vectors / distances (must be autograd-connected to coord) ---
        if ("diff" not in out) or ("dist" not in out):
            cell = out.get("cell", None)
            shift = out.get("shift", None)

            if cell is not None and shift is None:
                raise KeyError("PBC batch has 'cell' but missing 'shift' (expected from NL builder).")

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
    def _compute_diff_dist(
        self,
        coord: torch.Tensor,
        ind_2: torch.Tensor,
        cell: Optional[torch.Tensor],
        shift: Optional[torch.Tensor],
        ind_1: torch.Tensor,
    ):
        i = ind_2[:, 0]
        j = ind_2[:, 1]

        # No PBC info
        if cell is None or shift is None:
            diff = coord[j] - coord[i]
            dist = torch.linalg.norm(diff, dim=1)
            return diff, dist

        # PBC: use explicit image shifts; DO NOT wrap coords
        if cell.ndim == 2:
            H = cell.to(device=coord.device, dtype=coord.dtype)     # (3,3)
            t = shift.to(coord.dtype) @ H                           # (M,3)
            diff = (coord[j] + t) - coord[i]
            dist = torch.linalg.norm(diff, dim=1)
            return diff, dist

        if cell.ndim == 3:
            if ind_1.dtype != torch.long:
                ind_1 = ind_1.long()
            batch = ind_1[:, 0] if ind_1.ndim == 2 else ind_1       # (N,)

            sid = batch[i]                                          # (M,)
            H_pair = cell[sid].to(device=coord.device, dtype=coord.dtype)  # (M,3,3)
            t = torch.einsum("mi,mij->mj", shift.to(coord.dtype), H_pair)  # (M,3)
            diff = (coord[j] + t) - coord[i]
            dist = torch.linalg.norm(diff, dim=1)
            return diff, dist

        raise ValueError(f"Unexpected cell shape {tuple(cell.shape)}")

class PiNetTorch(nn.Module):
    """
    Full PiNet forward pass in PyTorch (dict-in, tensor-out), mirroring TF PiNet.call().

    Neighbor list behavior:
        - If tensors contains "cell", periodic MIC displacements are used.
        - Otherwise, free-space neighbor list is used.

    Current implementation uses a linked-cell (cell list) neighbor builder in
    PreprocessLayerTorch for performance, with brute-force methods kept as
    optional fallbacks during development.
    """

    def __init__(
        self,
        *,
        atom_types: Sequence[int] = (1, 6, 7, 8),
        rc: float = 4.0,
        cutoff_type: str = "f1",
        basis_type: str = "polynomial",
        n_basis: int = 4,
        gamma: float = 3.0,
        center: Optional[Sequence[float]] = None,
        pp_nodes: Sequence[int] = (16, 16),
        pi_nodes: Sequence[int] = (16, 16),
        ii_nodes: Sequence[int] = (16, 16),
        out_nodes: Sequence[int] = (16, 16),
        out_units: int = 1,
        out_pool: Union[bool, str] = False,
        act: str = "tanh",
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.depth = int(depth)
        self.preprocess = PreprocessLayerTorch(atom_types, rc)
        self.cutoff = CutoffFuncTorch(rc, cutoff_type)

        if basis_type == "polynomial":
            self.basis_fn = PolynomialBasisTorch(n_basis)
        elif basis_type == "gaussian":
            self.basis_fn = GaussianBasisTorch(center=center, gamma=gamma, rc=rc, n_basis=n_basis)
        else:
            raise ValueError(f"Unknown basis_type={basis_type!r}")

        # Mirror TF: first GCBlock has empty pp_nodes
        self.gc_blocks = nn.ModuleList(
            [GCBlockTorch([], pi_nodes, ii_nodes, n_basis=n_basis, activation=act)]
            + [
                GCBlockTorch(pp_nodes, pi_nodes, ii_nodes, n_basis=n_basis, activation=act)
                for _ in range(self.depth - 1)
            ]
        )
        self.out_layers = nn.ModuleList([OutLayerTorch(out_nodes, out_units, activation=act) for _ in range(self.depth)])
        self.res_updates = nn.ModuleList([ResUpdateTorch() for _ in range(self.depth)])
        self.ann_output = ANNOutputTorch(out_pool)

    def forward(self, tensors: dict) -> torch.Tensor:
        """
        Args:
            tensors: Dict with required keys:
              - ind_1, elems, coord
            Optional keys:
              - ind_2, dist, diff, prop (will be computed if missing)

        Returns:
            Output tensor:
              - per-atom if out_pool is falsy
              - per-structure if out_pool is a pooling mode
        """
        # At the top of forward()
        if not tensors.get("_preprocessed", False):
            tensors = self.preprocess(tensors)
        fc = self.cutoff(tensors["dist"])
        basis = self.basis_fn(tensors["dist"], fc=fc)

        output = torch.zeros((tensors["prop"].shape[0], self.out_layers[0].out_units),
                             dtype=tensors["prop"].dtype, device=tensors["prop"].device)

        for i in range(self.depth):
            prop_i = self.gc_blocks[i](tensors["ind_2"], tensors["prop"], basis)
            output = self.out_layers[i](tensors["ind_1"], prop_i, output)
            tensors["prop"] = self.res_updates[i](tensors["prop"], prop_i)

        output = self.ann_output(tensors["ind_1"], output)
        return output
    
class CellListNLPyTorch(nn.Module):
    """
    Linked-cell neighbor list builder (directed i->j pairs) with optional PBC.

    This is a performance-oriented replacement for the brute-force cdist neighbor
    list. It bins atoms into a grid of cells with linear size ~= rc, then checks
    only atoms in neighboring cells (27 neighbors in 3D).

    PBC rule (matches PiNN style):
        - If 'cell' is provided, we compute displacements using MIC in fractional
          coordinates and map back to Cartesian using the cell matrix convention:
              frac = coord @ inv(cell)
              coord = frac @ cell
        - If 'cell' is not provided, we treat the system as non-periodic.

    Notes:
        - Intended to be correct and much faster than O(n^2) for typical systems.
        - For very small systems, brute-force is fine; this is for scaling.
    """

    def __init__(self, rc: float) -> None:
        super().__init__()
        self.rc = float(rc)

        # 27 neighbor cell offsets in 3D
        offs = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    offs.append((dx, dy, dz))
        self.register_buffer("neighbor_offsets", torch.tensor(offs, dtype=torch.long))

    @torch.no_grad()
    def forward(self, coord: torch.Tensor, *, cell: Optional[torch.Tensor] = None) -> dict:
        """
        Build neighbor list for a single structure.

        Args:
            coord: (n,3) Cartesian coordinates.
            cell:  Optional (3,3) cell matrix. If provided, PBC/MIC is used.

        Returns:
            dict with:
              - ind_2: (n_pairs,2) long tensor of directed pairs (i,j)
        """
        device = coord.device
        dtype = coord.dtype
        rc = self.rc
        rc2 = rc * rc 

        n = int(coord.shape[0])
        if n == 0:
            return {
                "ind_2": torch.zeros((0, 2), dtype=torch.long, device=device),
                "shift": torch.zeros((0, 3), dtype=torch.long, device=device),
            }

        if cell is None:
            # Non-PBC: build a bounding box in Cartesian
            cmin = coord.min(dim=0).values
            cmax = coord.max(dim=0).values
            span = (cmax - cmin).clamp_min(1e-6)
            # cell size = rc
            ncell = torch.floor(span / rc).to(torch.long).clamp_min(1)
            # assign cell indices
            rel = (coord - cmin) / rc
            ci = torch.floor(rel).to(torch.long)
            ci = torch.minimum(ci, ncell - 1)
            # linearize
            mult = torch.tensor([ncell[1] * ncell[2], ncell[2], 1], device=device, dtype=torch.long)
            lin = (ci * mult).sum(dim=1)
            pbc = False
            H = None
            H_inv = None
        else:
            # PBC in fractional space: define grid in fractional coords
            H = cell.to(device=device, dtype=dtype)
            H_inv = torch.linalg.inv(H)
            frac = coord @ H_inv  # (n,3)
            frac = frac - torch.floor(frac)  # wrap into [0,1)
            # grid resolution: cells per axis ~ |a|/rc etc. in Cartesian is messy for triclinic;
            # in fractional space, we just choose ncell from effective lengths.
            # Use norms of cell vectors as approximate lengths.
            lengths = torch.linalg.norm(H, dim=1).clamp_min(1e-6)  # (3,)
            ncell = torch.floor(lengths / rc).to(torch.long).clamp_min(1)
            # mic_ok: MIC is valid only if cutoff sphere fits inside the cell;
            # otherwise we must enumerate periodic images (ASE-style).
            minL = float(lengths.min().item())
            mic_ok = (rc <= 0.5 * minL)

            shifts = None
            T = None

            # Only build extended-image shifts if MIC is NOT valid
            if not mic_ok:
                kx = int(np.ceil(rc / float(lengths[0].item())))
                ky = int(np.ceil(rc / float(lengths[1].item())))
                kz = int(np.ceil(rc / float(lengths[2].item())))

                sx = torch.arange(-kx, kx + 1, device=device, dtype=torch.long)
                sy = torch.arange(-ky, ky + 1, device=device, dtype=torch.long)
                sz = torch.arange(-kz, kz + 1, device=device, dtype=torch.long)
                Sx, Sy, Sz = torch.meshgrid(sx, sy, sz, indexing="ij")
                shifts = torch.stack([Sx.reshape(-1), Sy.reshape(-1), Sz.reshape(-1)], dim=1)  # (S,3)
                T = shifts.to(dtype) @ H  # (S,3) translation vectors in Cartesian

            rel = frac * ncell.to(dtype=dtype)
            ci = torch.floor(rel).to(torch.long)
            ci = torch.minimum(ci, ncell - 1)
            mult = torch.tensor([ncell[1] * ncell[2], ncell[2], 1], device=device, dtype=torch.long)
            lin = (ci * mult).sum(dim=1)
            pbc = True
            # --- fast-path: degenerate grid (all atoms in one cell) ---
            # Only meaningful when we are in the extended-image fallback (T/shifts exist).
            if (not mic_ok) and int(ncell.min().item()) == 1 and int(ncell.max().item()) == 1:
                ii, jj = torch.meshgrid(
                    torch.arange(n, device=device, dtype=torch.long),
                    torch.arange(n, device=device, dtype=torch.long),
                    indexing="ij",
                )
                ii = ii.reshape(-1)
                jj = jj.reshape(-1)

                di = coord[ii]   # (P,3)
                dj0 = coord[jj]  # (P,3)

                d = (dj0.unsqueeze(0) + T.unsqueeze(1)) - di.unsqueeze(0)  # (S,P,3)
                dist2 = (d * d).sum(dim=-1)                                # (S,P)
                keep = (dist2 > 0.0) & (dist2 < rc2)

                if keep.any():
                    s_idx, p_idx = keep.nonzero(as_tuple=True)
                    pairs = torch.stack([ii[p_idx], jj[p_idx]], dim=1)     # (K,2)
                    sh = shifts[s_idx]                                     # (K,3)
                    rows = torch.cat([pairs, sh], dim=1)                   # (K,5)
                    rows_uniq = torch.unique(rows, dim=0)
                    return {"ind_2": rows_uniq[:, :2], "shift": rows_uniq[:, 2:]}
                else:
                    return {
                        "ind_2": torch.zeros((0, 2), dtype=torch.long, device=device),
                        "shift": torch.zeros((0, 3), dtype=torch.long, device=device),
                    }
            

        # Build buckets: map linear cell id -> list of atom indices
        # We do this by sorting atoms by cell id, then slicing contiguous runs.
        order = torch.argsort(lin)
        lin_sorted = lin[order]

        # Find segment boundaries
        is_new = torch.ones_like(lin_sorted, dtype=torch.bool)
        is_new[1:] = lin_sorted[1:] != lin_sorted[:-1]
        starts = torch.nonzero(is_new, as_tuple=False).squeeze(1)
        ends = torch.cat([starts[1:], torch.tensor([n], device=device, dtype=starts.dtype)])

        unique_cells = lin_sorted[starts]  # cell ids present

        # Helper: convert linear id back to 3D cell index
        def unravel(lid: torch.Tensor) -> torch.Tensor:
            cx = lid // mult[0]
            rem = lid - cx * mult[0]
            cy = rem // mult[1]
            cz = rem - cy * mult[1]
            return torch.stack([cx, cy, cz], dim=-1)

        cell_xyz = unravel(unique_cells)  # (n_occ,3)

        # Build a dict-like lookup from cell xyz -> segment [start,end)
        # We'll use a hash: key = cx*(ny*nz)+cy*nz+cz == linear id, so unique_cells already is key.
        # We need to lookup neighbor cell ids quickly: we can binary-search in unique_cells.
        # Since unique_cells is sorted, torch.searchsorted works.
        ind2_list = []
        shift_list = []

        # Iterate occupied cells
        for k in range(unique_cells.numel()):
            lid = unique_cells[k]
            a0, a1 = int(starts[k].item()), int(ends[k].item())
            atoms_i = order[a0:a1]  # atoms in this cell

            cxyz = cell_xyz[k]  # (3,)
            # 27 neighbor cells
            neigh_xyz = cxyz[None, :] + self.neighbor_offsets  # (27,3)
            if pbc:
                neigh_xyz = neigh_xyz % ncell[None, :]  # wrap cell indices
            else:
                # clamp for non-pbc: drop out-of-range neighbor cells
                inb = (neigh_xyz >= 0).all(dim=1) & (neigh_xyz < ncell[None, :]).all(dim=1)
                neigh_xyz = neigh_xyz[inb]

            # Convert neighbor xyz -> linear ids
            neigh_lid = (neigh_xyz * mult[None, :]).sum(dim=1)
            # Find which neighbor cells are occupied
            pos = torch.searchsorted(unique_cells, neigh_lid)

            in_range = pos < unique_cells.numel()
            pos_in = pos[in_range]
            neigh_in = neigh_lid[in_range]

            ok = unique_cells[pos_in] == neigh_in
            pos = pos_in[ok]

            for kk in pos.tolist():
                b0, b1 = int(starts[kk].item()), int(ends[kk].item())
                atoms_j = order[b0:b1]
                if atoms_i.numel() == 0 or atoms_j.numel() == 0:
                    continue

                # Generate all cross pairs between atoms_i and atoms_j
                ii = atoms_i.repeat_interleave(atoms_j.numel())
                jj = atoms_j.repeat(atoms_i.numel())

                # Remove self-pairs only in non-PBC case.
                # In PBC, we must allow i==j because self-images (shift != 0) are real neighbors
                # and ASE neighbor_list includes them.
                if not pbc:
                    mask = ii != jj
                    if mask.any():
                        ii = ii[mask]
                        jj = jj[mask]
                    else:
                        continue  

                # Build candidate pairs (ii, jj). Filter by distance without building graph.
                # It's fine to compute distances here under no_grad just for pruning pairs.
                if not pbc:
                    d = coord[jj] - coord[ii]
                    dist2 = (d * d).sum(dim=-1)
                    keep = (dist2 > 0.0) & (dist2 < rc2)
                    if keep.any():
                        ind2_list.append(torch.stack([ii[keep], jj[keep]], dim=1))
                        shift_list.append(torch.zeros((int(keep.sum().item()), 3), dtype=torch.long, device=device))
                else:
                    if mic_ok:
                        # --- FAST PATH: MIC per pair (no image enumeration) ---
                        di_frac = frac[ii]  # (P,3)
                        dj_frac = frac[jj]  # (P,3)

                        dfrac = dj_frac - di_frac                    # (P,3)
                        shift = -torch.round(dfrac).to(torch.long)    # (P,3) integer MIC shift
                        dfrac = dfrac + shift.to(dfrac.dtype)         # wrap into [-0.5, 0.5] (MIC)

                        d = dfrac @ H                                 # (P,3) Cartesian displacement
                        dist2 = (d * d).sum(dim=-1)                   # (P,)
                        keep = (dist2 > 0.0) & (dist2 < rc2)

                        if keep.any():
                            ind2_list.append(torch.stack([ii[keep], jj[keep]], dim=1))
                            shift_list.append(shift[keep])

                    else:
                        # --- FALLBACK: extended images (physics when rc is large vs cell) ---
                        di = coord[ii]    # (P,3)
                        dj0 = coord[jj]   # (P,3)

                        d = (dj0.unsqueeze(0) + T.unsqueeze(1)) - di.unsqueeze(0)  # (S,P,3)
                        dist2 = (d * d).sum(dim=-1)                                # (S,P)
                        keep = (dist2 > 0.0) & (dist2 < rc2)

                        if keep.any():
                            s_idx, p_idx = keep.nonzero(as_tuple=True)
                            pairs = torch.stack([ii[p_idx], jj[p_idx]], dim=1)     # (K,2)
                            sh = shifts[s_idx]                                     # (K,3)
                            ind2_list.append(pairs)
                            shift_list.append(sh)


        if len(ind2_list) == 0:
            return {
                "ind_2": torch.zeros((0, 2), dtype=torch.long, device=device),
                "shift": torch.zeros((0, 3), dtype=torch.long, device=device),
            }

        ind_2 = torch.cat(ind2_list, dim=0).long()
        shift = torch.cat(shift_list, dim=0).long()

        rows = torch.cat([ind_2, shift], dim=1)  # (M,5) = (i, j, sx, sy, sz)
        rows_uniq = torch.unique(rows, dim=0)

        return {"ind_2": rows_uniq[:, :2], "shift": rows_uniq[:, 2:]}