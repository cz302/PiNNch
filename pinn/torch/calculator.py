# pinn/torch/calculator.py
from __future__ import annotations

import os
import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from pinn.torch.device import resolve_device

from .model import get_model as build_model

def _get_stress_mode(model) -> str:
    mode = _find_attr_in_wrapped_model(model, "stress_mode", "diff")
    return str(mode).lower()

def _get_fd_eps(model) -> float:
    return float(_find_attr_in_wrapped_model(model, "fd_eps", 1e-4))


def _find_attr_in_wrapped_model(obj, attr: str, default=None):
    """
    Walk common wrapper attributes/children to find attr on the underlying torch network.
    Returns default if not found.
    """
    seen = set()
    stack = [obj]
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))

        if hasattr(cur, attr):
            return getattr(cur, attr)

        # common wrappers
        for name in ("network", "net", "module", "model"):
            nxt = getattr(cur, name, None)
            if nxt is not None and nxt is not cur:
                stack.append(nxt)

        # torch module children
        try:
            import torch.nn as nn
            if isinstance(cur, nn.Module):
                stack.extend(list(cur.children()))
        except Exception:
            pass

    return default


def _find_preprocess(obj):
    """
    Try a bunch of common wrapper attributes to find a .preprocess(tensors) callable.
    Returns callable or None.
    """
    # direct
    if hasattr(obj, "preprocess") and callable(getattr(obj, "preprocess")):
        return obj.preprocess

    # common wrappers
    for attr in ("network", "net", "module", "model"):
        if hasattr(obj, attr):
            sub = getattr(obj, attr)
            if hasattr(sub, "preprocess") and callable(getattr(sub, "preprocess")):
                return sub.preprocess

    return None

class TorchPiNNCalc(Calculator):
    """ASE calculator for the Torch backend.

    Computes:
      - energy: from model(tensors)
      - forces: -dE/dR via autograd
      - stress: virial-based, ASE 6-vector convention
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, model, *, device="auto", virial_mode="diff", to_eV: float = 1.0, **kwargs):
        """ASE calculator for Torch PiNN.

        Args:
            model: Torch potential model. Its native output units can be arbitrary.
            device: "auto"|"cpu"|"cuda"|...
            virial_mode: stress mode selection (kept as-is).
            to_eV: Multiplicative conversion factor from the model's native *energy* units
                to eV. Forces and stresses are scaled consistently because they are
                derivatives of energy.
                Example: kcal/mol -> eV: to_eV = 0.0433641153088
        """
        super().__init__(**kwargs)
        dev = resolve_device(device)
        self.device = str(dev)
        self.model = model.to(self.device)
        self.virial_mode = virial_mode
        self.to_eV = float(to_eV)

    def calculate(self, atoms=None, properties=("energy", "forces", "stress"), system_changes=all_changes):
        """Run torch model and populate ASE results dict.

        Contract:
        - Build PiNN sparse tensors dict (coord/elems/ind_1/(cell)).
        - Call model(tensors) -> energy per structure, shape (B,). Here B=1.
        - Forces from autograd: F = -dE/dR (R is tensors["coord"]).
        - Apply e_dress (in model energy units) then multiply by e_unit.
        """
        super().calculate(atoms, properties, system_changes)

        R_np = atoms.get_positions().astype(np.float32)  # (N,3)
        Z_np = atoms.numbers.astype(np.int64)            # (N,)
        N = R_np.shape[0]

        pbc = np.any(atoms.get_pbc())
        need_stress = ("stress" in properties) and pbc

     
        torsion_boost = bool(_find_attr_in_wrapped_model(self.model, "torsion_boost", False))
        stress_mode = _get_stress_mode(self.model)  # dist|diff|cell|fd (user-controlled)
        fd_eps = _get_fd_eps(self.model)

        # --- Build tensors ---
        elems = torch.tensor(Z_np, dtype=torch.long, device=self.device)
        ind_1 = torch.zeros((N, 1), dtype=torch.long, device=self.device)

        if pbc:
            # Cell tensor (leaf only if we will use thermo stress)
            cell = torch.tensor(atoms.cell.array, dtype=torch.float32, device=self.device)
            if need_stress and torsion_boost:
                cell = cell.detach().clone().requires_grad_(True)

            # Fractional coordinates as leaf (kept fixed under scale_atoms=True)
            frac = torch.tensor(
                atoms.get_scaled_positions(wrap=False),
                dtype=torch.float32,
                device=self.device,
            ).detach().clone().requires_grad_(True)

            # Cartesian coords derived from frac and cell (DO NOT detach; must keep dependency on cell)
            coord = frac @ cell
        else:
            cell = None
            frac = None
            coord = torch.tensor(R_np, dtype=torch.float32, device=self.device).detach().clone().requires_grad_(True)

        tensors = {
            "coord": coord,  # note: under PBC this is not a leaf, and that's OK
            "elems": elems,
            "ind_1": ind_1,
        }
        if pbc:
            tensors["cell"] = cell

        preprocess = _find_preprocess(self.model)
        if pbc and preprocess is not None:
            coord0 = tensors["coord"]
            cell0  = tensors["cell"]
            # We need preprocess if:
            # - stress_mode is dist/diff (needs dist/diff/ind_2)
            # - stress_mode is cell/fd (energy still depends on neighbor features)
            # In practice: preprocess once for all PBC paths.
            tensors = preprocess(tensors)
            tensors["_preprocessed"] = True
            # Identity checks: did preprocess replace the Tensor object?
            assert tensors["coord"] is coord0, "preprocess overwrote coord"
            assert tensors["cell"]  is cell0,  "preprocess overwrote cell"
        

        # Model already returns total energy in ASE units
        E = self.model(tensors)
        if self.to_eV != 1.0:
            E = E * self.to_eV

        if E.ndim == 0:
            E = E.view(1)

        if E.numel() != 1:
            raise ValueError(
                f"Calculator expects one structure energy; got shape {tuple(E.shape)}"
            )
        # --- Forces from autograd on coordinates ---
        if pbc:
            (dE_dfrac,) = torch.autograd.grad(
                E.sum(), frac,
                create_graph=False,
                retain_graph=need_stress,
            )
            inv_cell_T = torch.inverse(cell).transpose(0, 1)
            dE_dR = dE_dfrac @ inv_cell_T
            F = -dE_dR
            assert torch.isfinite(dE_dfrac).all()
        else:
            (dE_dR,) = torch.autograd.grad(
                E.sum(), coord,
                create_graph=False,
                retain_graph=need_stress,
            )
            F = -dE_dR
            assert torch.isfinite(dE_dR).all()

        self.results["energy"] = float(E.item())
        self.results["forces"] = F.detach().cpu().numpy()

        # --- Stress (PBC only, if requested) ---
        if need_stress:
            diff = tensors.get("diff", None)   # (n_pairs,3)
            dist = tensors.get("dist", None)   # (n_pairs,)
            ind_2 = tensors.get("ind_2", None) # (n_pairs,2)
            cell = tensors.get("cell", None)   # (3,3)

            if cell is None:
                raise RuntimeError("PBC stress requires tensors['cell'].")

            V = torch.det(cell).abs().clamp_min(1e-12)

            def to_ase_stress6(sigma_3x3: torch.Tensor) -> np.ndarray:
                # sigma row-major -> ASE Voigt [xx, yy, zz, yz, xz, xy]
                s = sigma_3x3.reshape(-1)
                return np.array(
                    [s[0].item(), s[4].item(), s[8].item(), s[5].item(), s[2].item(), s[1].item()],
                    dtype=float,
                )

            if stress_mode in ("dist", "diff"):
                if diff is None or ind_2 is None:
                    raise RuntimeError("dist/diff stress requires tensors['diff'] and tensors['ind_2'].")

                if stress_mode == "diff":
                    dE_ddiff = torch.autograd.grad(E.sum(), diff, create_graph=False, retain_graph=False)[0]  # (n_pairs,3)
                else:
                    if dist is None:
                        raise RuntimeError("dist stress requires tensors['dist'].")
                    dE_ddist = torch.autograd.grad(E.sum(), dist, create_graph=False, retain_graph=False)[0]  # (n_pairs,)
                    inv_r = dist.clamp_min(1e-12).reciprocal()
                    dE_ddiff = dE_ddist.unsqueeze(-1) * diff * inv_r.unsqueeze(-1)  # chain rule

                # Pair virial tensor contribution: outer(diff, dE_ddiff)
                # Summing over directed pairs is fine if TF did the same; otherwise you may need 0.5.
                outer = diff.unsqueeze(2) * dE_ddiff.unsqueeze(1)  # (n_pairs,3,3)
                sigma = outer.sum(dim=0) / V                       # (3,3)
                self.results["stress"] = to_ase_stress6(sigma)
                return

            if stress_mode == "cell":
                # Full thermodynamic stress at fixed fractional coords:
                # sigma = -(dE/dcell @ cell^T) / V
                dE_dcell = torch.autograd.grad(E.sum(), cell, create_graph=False, retain_graph=False)[0]  # (3,3)
                sigma = -(dE_dcell @ cell.transpose(0, 1)) / V
                self.results["stress"] = to_ase_stress6(sigma)
                return

            if stress_mode == "fd":
                # Finite-difference stress by small symmetric strain on cell (fractional coords fixed).
                # sigma_ij = (1/V) * dE/dε_ij, ASE uses opposite sign convention for "pressure"
                # so we follow the same sigma -> stress6 mapping as above.
                eps = float(fd_eps)

                # We need fractional coords as a leaf to keep them fixed:
                frac = None
                # Best effort: reconstruct from atoms each call
                frac = torch.tensor(
                    atoms.get_scaled_positions(wrap=False),
                    dtype=torch.float32,
                    device=self.device,
                )

                def energy_for_cell(cell_new: torch.Tensor) -> torch.Tensor:
                    # Build tensors with fixed frac and new cell
                    coord_new = frac @ cell_new
                    t = {
                        "coord": coord_new,
                        "elems": elems,
                        "ind_1": ind_1,
                        "cell": cell_new,
                    }
                    preprocess = _find_preprocess(self.model)
                    if preprocess is None:
                        raise RuntimeError("fd stress requires preprocess().")
                    t = preprocess(t)
                    t["_preprocessed"] = True
                    E_new = self.model(t)
                    if self.to_eV != 1.0:
                         E_new = E_new * self.to_eV
                    return E_new.view(-1).sum()

                # Symmetric strain basis for Voigt order: xx, yy, zz, yz, xz, xy
                # We perturb the cell by right-multiplying with (I + ε * S)
                I = torch.eye(3, dtype=cell.dtype, device=cell.device)

                S_list = []
                # xx
                S = torch.zeros_like(I); S[0, 0] = 1.0; S_list.append(S)
                # yy
                S = torch.zeros_like(I); S[1, 1] = 1.0; S_list.append(S)
                # zz
                S = torch.zeros_like(I); S[2, 2] = 1.0; S_list.append(S)
                # yz
                S = torch.zeros_like(I); S[1, 2] = 0.5; S[2, 1] = 0.5; S_list.append(S)
                # xz
                S = torch.zeros_like(I); S[0, 2] = 0.5; S[2, 0] = 0.5; S_list.append(S)
                # xy
                S = torch.zeros_like(I); S[0, 1] = 0.5; S[1, 0] = 0.5; S_list.append(S)

                dE_dstrain = []
                for S in S_list:
                    cell_p = cell @ (I + eps * S)
                    cell_m = cell @ (I - eps * S)
                    Ep = energy_for_cell(cell_p)
                    Em = energy_for_cell(cell_m)
                    dE = (Ep - Em) / (2.0 * eps)
                    dE_dstrain.append(dE)

                # Assemble sigma in Voigt back to tensor (symmetric)
                sxx, syy, szz, syz, sxz, sxy = dE_dstrain
                sigma = torch.zeros((3, 3), dtype=cell.dtype, device=cell.device)
                sigma[0, 0] = sxx / V
                sigma[1, 1] = syy / V
                sigma[2, 2] = szz / V
                sigma[1, 2] = sigma[2, 1] = syz / V
                sigma[0, 2] = sigma[2, 0] = sxz / V
                sigma[0, 1] = sigma[1, 0] = sxy / V

                self.results["stress"] = to_ase_stress6(sigma)
                return

            raise ValueError(f"Unknown stress_mode={stress_mode!r}")
        


def get_calc(model_spec, **kwargs):
    """Factory used by pinn.get_calc() for backend='torch'."""

    # --- Special-case analytic LJ: no checkpoint, just return ASE's LJ calc ---
    if model_spec.get("network", {}).get("name") == "LJ":
        from ase.calculators.lj import LennardJones
        rc = float(model_spec.get("network", {}).get("params", {}).get("rc", 3.0))
        return LennardJones(rc=rc)

    model_dir = model_spec.get("model_dir", None)
    if model_dir is None:
        raise ValueError("Torch calculator requires model_spec['model_dir'].")

    ckpt_path = os.path.join(model_dir, "model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing torch checkpoint: {ckpt_path}")

    device = kwargs.pop("device", "auto")
    dev = resolve_device(device)
    device = str(dev)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Prefer rebuilding from checkpoint params (most reliable)
    ckpt_params = ckpt.get("params", None)
    build_params = ckpt_params if isinstance(ckpt_params, dict) else model_spec

    model = build_model(build_params).to(device)
    
    # Support both formats: full dict checkpoint or raw state_dict
    state = ckpt.get("model_state_dict", ckpt)

    # ... existing code that builds `model` and loads `state` from model.pt ...

    strict = bool(kwargs.pop("strict", True))

    # Existing line (currently strict=True implicitly):
    # model.load_state_dict(state)

    missing, unexpected = model.load_state_dict(state, strict=strict)

    if (missing or unexpected):
        # Keep the existing warning style you already use elsewhere in the repo
        print("[checkpoint] Loaded into model:")
        print(f"  missing keys   = {len(missing)}")
        print(f"  unexpected keys= {len(unexpected)}")
        if missing:
            print("  missing (first 10):", missing[:10])
        if unexpected:
            print("  unexpected (first 10):", unexpected[:10])

    model.eval()

    virial_mode = (
        model_spec.get("model", {})
                .get("params", {})
                .get("virial_mode", "diff")
    )

    return TorchPiNNCalc(model, device=device, virial_mode=virial_mode, **kwargs)
