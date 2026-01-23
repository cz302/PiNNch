# pinn/torch/model.py
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from pinn.networks.pinet_torch import PiNetTorch
from pinn.networks.pinet2_torch import PiNet2Torch


class PiNetPotentialTorch(nn.Module):
    def __init__(self, net, *, e_dress, e_scale, e_unit):
        super().__init__()
        self.net = net
        self.e_dress = {int(k): float(v) for k, v in (e_dress or {}).items()}
        self.e_scale = float(e_scale)
        self.e_unit = float(e_unit)

    def _sum_to_struct(self, x_atom: torch.Tensor, ind_1: torch.Tensor | None) -> torch.Tensor:
        if ind_1 is None:
            return x_atom.sum().view(1)
        if ind_1.dtype != torch.long:
            ind_1 = ind_1.long()
        sid = ind_1[:, 0]
        B = int(sid.max().item()) + 1 if sid.numel() > 0 else 1
        out = x_atom.new_zeros((B,))
        out.index_add_(0, sid, x_atom)
        return out

    def dress_struct(self, tensors: dict) -> torch.Tensor:
        """
        Per-structure dressing energy in RAW label units (Hartree).
        Shape: (B,) or (1,)
        """
        ind_1 = tensors.get("ind_1", None)
        like = tensors["coord"]  # just for dtype/device
        if not self.e_dress:
            # return zeros with correct shape/device/dtype
            if ind_1 is None:
                return like.new_zeros((1,), dtype=like.dtype)
            if ind_1.dtype != torch.long:
                ind_1 = ind_1.long()
            sid = ind_1[:, 0]
            B = int(sid.max().item()) + 1 if sid.numel() > 0 else 1
            return like.new_zeros((B,), dtype=like.dtype)

        elems = tensors["elems"]
        dress_atoms = like.new_zeros((elems.shape[0],), dtype=like.dtype)
        for Z, val in self.e_dress.items():
            mask = (elems == int(Z))
            if mask.any():
                dress_atoms[mask] = float(val)

        return self._sum_to_struct(dress_atoms, ind_1)

    def forward_train(self, tensors: dict) -> torch.Tensor:
        """
        TRAIN/EVAL forward: returns E_pred_scaled in *scaled label space*.
        (No unit conversion, no dressing, no undo scale)
        """
        ann_out = self.net(tensors)
        e_atom = ann_out["energy"] if isinstance(ann_out, dict) else ann_out
        return self._sum_to_struct(e_atom, tensors.get("ind_1", None))

    def forward(self, tensors: dict) -> torch.Tensor:
        """
        PREDICT forward: returns E_out in eV (TF-PREDICT semantics).
        """
        E_scaled = self.forward_train(tensors)
        dress = self.dress_struct(tensors)          # RAW 
        E_raw = E_scaled / self.e_scale + dress     # RAW
        E_out = E_raw * self.e_unit                 # eV
        return E_out


def _materialize_lazy(model: torch.nn.Module, *, atom_types) -> None:
    """Run a tiny dummy forward to materialize LazyLinear parameters.

    The calc-reload smoke test saves model.state_dict() immediately after
    pinn.get_model(params), so Lazy modules must be initialized here.
    """
    # Put dummy tensors on the same device as the model (cpu in tests, but be safe).
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    z0 = int(atom_types[0]) if len(atom_types) else 1

    tensors = {
        "coord": torch.zeros((2, 3), dtype=torch.float32, device=device),              # (N,3)
        "elems": torch.tensor([z0, z0], dtype=torch.long, device=device),              # (N,)
        "ind_1": torch.zeros((2, 1), dtype=torch.long, device=device),                 # (N,1)
    }

    model.eval()
    with torch.no_grad():
        _ = model.forward_train(tensors)

def get_model(params: dict, **kwargs) -> nn.Module:
    """Torch backend model factory.

    Expected schema (mirrors TF tests):
      params["network"]["params"]: PiNet hyperparameters
      params["model"]["params"]:   e_dress, e_scale, e_unit, use_force, etc.
    """
    net_params = params["network"]["params"]
    mparams = params["model"]["params"]

    # Require the keys that define the architecture in tests/YAML.
    required = ["atom_types", "rc"]
    missing = [k for k in required if k not in net_params]
    if missing:
        raise KeyError(f"Missing network params for torch PiNet: {missing}")

    # Optional / nice-to-have defaults are OK.
    act = net_params.get("act", "tanh")

    net_name = params["network"]["name"]

    common_kwargs = dict(
        atom_types=net_params["atom_types"],
        rc=float(net_params["rc"]),
        n_basis=int(net_params.get("n_basis", 5)),
        depth=int(net_params.get("depth", 3)),
        pp_nodes=net_params.get("pp_nodes", [8, 8]),
        pi_nodes=net_params.get("pi_nodes", [8, 8]),
        ii_nodes=net_params.get("ii_nodes", [8, 8]),
        out_nodes=net_params.get("out_nodes", [8, 8]),
        act=net_params.get("act", "tanh"),
        out_units=1,
        out_pool=False,
    )

    if net_name == "PiNet":
        net = PiNetTorch(**common_kwargs)

    elif net_name == "PiNet2":
        """Build PiNet2Torch from network params.

        PiNet2Torch takes a single params dict (unlike PiNetTorch which uses kwargs).
        We therefore forward common PiNet params plus PiNet2-specific add-ons.
        """
        pinet2_params = dict(common_kwargs)

        # Forward PiNet2-specific knobs (including optional feature add-ons)
        pinet2_params["rank"] = int(net_params.get("rank", 3))
        pinet2_params["torsion_boost"] = bool(net_params.get("torsion_boost", False))

        # Debug / test instrumentation (safe no-op in production)
        if "debug_tensors" in net_params:
            pinet2_params["debug_tensors"] = bool(net_params["debug_tensors"])

        # Forward other PiNet2 geometry/basis knobs if provided
        for k in ("basis_type", "cutoff_type", "gamma", "center", "out_units", "out_pool"):
            if k in net_params:
                pinet2_params[k] = net_params[k]

        net = PiNet2Torch(pinet2_params)
    else:
        raise ValueError(f"Unknown network: {net_name}")


    model = PiNetPotentialTorch(
        net,
        e_dress=mparams.get("e_dress", {}),
        e_scale=float(mparams.get("e_scale", 1.0)),
        e_unit=float(mparams.get("e_unit", 1.0)),
    )

    # --- Stress/virial configuration for Torch calculator ---
    model.stress_mode = str(mparams.get("stress_mode", "dist")).lower()
    model.fd_eps = float(mparams.get("fd_eps", 1e-4))

    _materialize_lazy(model, atom_types=net_params["atom_types"])

    return model
