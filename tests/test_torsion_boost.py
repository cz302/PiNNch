import pytest
import numpy as np
import torch
from ase import Atoms

import pinn


def _make_params(tmp_path, torsion_boost: bool):
    network_params = {
        "ii_nodes": [8, 8],
        "pi_nodes": [8, 8],
        "pp_nodes": [8, 8],
        "out_nodes": [8, 8],
        "depth": 3,
        "rc": 5.0,
        "n_basis": 5,
        "atom_types": [1],
        "rank": 3,
        "torsion_boost": torsion_boost,
    }
    return {
        "model_dir": str(tmp_path / ("tb_on" if torsion_boost else "tb_off")),
        "network": {"name": "PiNet2", "params": network_params},
        "model": {
            "name": "potential_model",
            "params": {
                "use_force": True,
                "e_dress": {1: 0.5},
                "e_scale": 5.0,
                "e_unit": 2.0,
            },
        },
    }


def _get_lj_like_data():
    # mirror your test_potential.py H3 dataset shape
    atoms = Atoms("H3", positions=[[0, 0, 0], [0, 1, 0], [1, 1, 0]])
    coord = []
    elems = []
    # keep it tiny: just a handful of frames is enough to write a checkpoint
    for x_a in np.linspace(-5, 0, 20):
        atoms.positions[0, 0] = x_a
        coord.append(atoms.positions.copy())
        elems.append(atoms.numbers)

    # dummy labels (training is just to produce model.pt; accuracy irrelevant here)
    # shapes must match what pinn.train_and_evaluate expects in torch path
    e_data = np.zeros((len(coord),), dtype=np.float64)
    f_data = np.zeros((len(coord), 3, 3), dtype=np.float64)

    return {
        "coord": np.array(coord),
        "elems": np.array(elems),
        "e_data": e_data,
        "f_data": f_data,
    }


def _autograd_forces_from_network(calc, atoms: Atoms):
    """
    Compute forces as -grad(E) directly from the same torch network that the calculator uses.
    For this to work, the calculator must expose the loaded model (recommended as calc.model).
    """
    # Make sure calc is initialized and has loaded model
    calc.calculate(atoms)

    model = getattr(calc, "model", None) or getattr(calc, "_model", None) or getattr(calc, "torch_model", None)
    assert model is not None, (
        "Calculator doesn't expose the underlying torch model. "
        "Expose it as `self.model` inside your torch ASE calculator."
    )
    net = getattr(model, "network", None) or getattr(model, "net", None) or getattr(model, "module", None) or model

    coord = torch.tensor(atoms.get_positions(), dtype=torch.float32, requires_grad=True)
    elems = torch.tensor(atoms.numbers, dtype=torch.long)
    ind_1 = torch.zeros((coord.shape[0],), dtype=torch.long)

    sample = {"coord": coord, "elems": elems, "ind_1": ind_1}

    e_pa = net(sample)          # (n_atoms, out_units) or (n_atoms,)
    e = e_pa.sum()

    (grad,) = torch.autograd.grad(e, coord, create_graph=False, retain_graph=False)
    f_ag = (-grad).detach().cpu().numpy()
    return f_ag


@pytest.mark.forked
@pytest.mark.parametrize("torsion_boost", [False, True])
def test_torch_forces_are_energy_gradient(tmp_path, monkeypatch, torsion_boost):
    monkeypatch.setenv("PINN_BACKEND", "torch")
    params = _make_params(tmp_path, torsion_boost=torsion_boost)

    # 1) Minimal training just to create model.pt so pinn.get_calc can load
    model = pinn.get_model(params)
    data = _get_lj_like_data()

    _ = pinn.train_and_evaluate(
        model=model,
        params=params,
        data=data,
        max_steps=5,          # tiny
        eval_steps=1,         # tiny
        batch_size_train=10,
        batch_size_eval=10,
        shuffle_buffer=20,
    )

    # 2) Now get_calc should succeed (model.pt exists)
    atoms = Atoms("H3", positions=[[0, 0, 0], [0, 1, 0], [1, 1, 0]])
    calc = pinn.get_calc(params, properties=["energy", "forces"])
    atoms.calc = calc

    # 3) Compare calculator forces vs autograd forces from same network
    f_calc = atoms.get_forces()
    f_ag = _autograd_forces_from_network(calc, atoms)
    # Match calculator units: calculator returns forces in "physical" units.
    # Autograd helper typically differentiates the *model-unit* energy.
    e_unit = float(params["model"]["params"].get("e_unit", 1.0))
    e_scale = float(params["model"]["params"].get("e_scale", 1.0))
    f_ag = f_ag * (e_unit / e_scale)

    assert np.isfinite(f_calc).all()
    assert np.isfinite(f_ag).all()

    assert np.allclose(f_calc, f_ag, rtol=1e-5, atol=1e-6), (
        f"Calculator forces are not -grad(E) for torsion_boost={torsion_boost}.\n"
        f"max|diff| = {np.max(np.abs(f_calc - f_ag))}"
    )
