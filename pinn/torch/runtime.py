
import os
import numpy as np
import torch
from typing import Dict, Iterator, Optional
from pinn.torch.optim import build_optimizer_from_params, apply_grad_clipping
from pinn.torch.input_pipeline import TorchDataOptions, _iter_batches_from_examples
from pinn.utils import init_params
import yaml
from torch.utils.tensorboard import SummaryWriter
import glob
from pathlib import Path
from pinn.io.build_dataset import build_dataset, BuildOptions
from pinn.torch.device import resolve_device

from dataclasses import dataclass

@dataclass(frozen=True)
class PotentialLossConfig:
    """Loss + reporting configuration for potential_model training."""
    e_unit: float = 1.0
    e_scale: float = 1.0
    e_loss_multiplier: float = 1.0
    f_loss_multiplier: float = 1.0
    use_force: bool = True
    use_e_per_atom: bool = False
    log_e_per_atom: bool = False  # affects metrics/reporting only


def maybe_compile_model(model: torch.nn.Module) -> torch.nn.Module:
    """Optionally compile the model for speed (PyTorch 2+)."""
    if os.environ.get("PINN_TORCH_COMPILE", "0") != "1":
        return model
    return torch.compile(model, mode="max-autotune")

def move_pinn_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out

def _ckpt_dir(model_dir: str) -> str:
    return os.path.join(model_dir, "checkpoints")


def _save_checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    params: dict,
    step: int,
    model_dir: str,
    keep_last: int = 5,
) -> str:
    os.makedirs(_ckpt_dir(model_dir), exist_ok=True)

    to_save = model.module if hasattr(model, "module") else model

    ckpt = {
        "step": int(step),
        "model_state_dict": to_save.state_dict(),
        "optimizer_state_dict": (optimizer.state_dict() if optimizer is not None else None),
        "scheduler_state_dict": (scheduler.state_dict() if scheduler is not None else None),
        "params": params,
        "pytorch_version": torch.__version__,
    }

    path = os.path.join(_ckpt_dir(model_dir), f"ckpt_step_{int(step):09d}.pt")
    torch.save(ckpt, path)

    # Update "latest" pointer
    latest_path = os.path.join(_ckpt_dir(model_dir), "latest.pt")
    torch.save(ckpt, latest_path)

    # Garbage-collect older checkpoints
    ckpts = sorted(glob.glob(os.path.join(_ckpt_dir(model_dir), "ckpt_step_*.pt")))
    if keep_last is not None and keep_last > 0 and len(ckpts) > keep_last:
        for p in ckpts[:-keep_last]:
            try:
                os.remove(p)
            except OSError:
                pass

    return path


def _load_latest_checkpoint(model_dir: str) -> Optional[dict]:
    latest = os.path.join(_ckpt_dir(model_dir), "latest.pt")
    if os.path.isfile(latest):
        return torch.load(latest, map_location="cpu")

    # Fallback: pick newest ckpt_step_*.pt
    ckpts = sorted(glob.glob(os.path.join(_ckpt_dir(model_dir), "ckpt_step_*.pt")))
    if ckpts:
        return torch.load(ckpts[-1], map_location="cpu")
    return None


def _try_resume_from_checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    model_dir: str,
    device: str,
) -> int:
    ckpt = _load_latest_checkpoint(model_dir)
    if ckpt is None:
        return 0

    to_load = model.module if hasattr(model, "module") else model
    to_load.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    # Move optimizer states to the chosen device (important!)
    if optimizer is not None:
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    # Return next step to run
    last_step = int(ckpt.get("step", 0))
    return last_step + 1

def _flatten_forces(F: torch.Tensor) -> torch.Tensor:
    # Accept (B,N,3) or (BN,3) and return (BN,3)
    if F.ndim == 3 and F.shape[-1] == 3:
        return F.reshape(-1, 3)
    if F.ndim == 2 and F.shape[-1] == 3:
        return F
    raise ValueError(f"Unexpected force shape: {tuple(F.shape)}")


def _inject_e_dress_into_torch_model(model, params: dict) -> None:
    mp = params.get("model", {}).get("params", {}) or {}
    dress = mp.get("e_dress", None)
    if not dress:
        return

    # YAML may load keys as int already, but be strict.
    dress = {int(k): float(v) for k, v in dress.items()}

    # Handle common wrappers: model, model.model, model.network, etc.
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(model, "network", None),
        getattr(model, "net", None),
    ]
    for obj in candidates:
        if obj is None:
            continue
        if hasattr(obj, "e_dress"):
            obj.e_dress = dress

def _structure_iter_from_torch_batches(batch_iter, max_structures=1000):
    """
    Yield per-structure dicts compatible with pinn.utils.init_params.

    This strips batching, gradients, devices, and sparse flattening.
    """
    seen = 0
    for batch in batch_iter:
        # YAML pipeline: coord is (B*N,3), ind_1 maps atoms -> structure
        coord = batch["coord"].detach().cpu()
        elems = batch["elems"].detach().cpu()
        e_data = batch["e_data"].detach().cpu()
        ind_1 = batch["ind_1"][:, 0].detach().cpu()

        B = int(e_data.shape[0])

        for b in range(B):
            mask = ind_1 == b
            yield {
                "coord": coord[mask].numpy(),
                "elems": elems[mask].numpy(),
                "e_data": float(e_data[b]),
            }
            seen += 1
            if seen >= max_structures:
                return

def _torch_structure_iter_to_tf_dataset(struct_iter):
    """
    Adapt a Torch/Python structure iterator to a tf.data.Dataset
    compatible with pinn.utils.init_params.
    """
    import tensorflow as tf

    def gen():
        for s in struct_iter:
            yield {
                "coord": s["coord"],
                "elems": s["elems"],
                "e_data": s["e_data"],
            }

    return tf.data.Dataset.from_generator(
        gen,
        output_signature={
            "coord": tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            "elems": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "e_data": tf.TensorSpec(shape=(), dtype=tf.float32),
        },
    )

def _get_nl_builder_from_model(model):
    """Return the neighbor-list builder module used by the Torch network.

    We keep this in one place because different wrappers may expose the network
    differently (model.network, model.net, model.model, etc.).

    Raises:
        AttributeError if no builder can be found.
    """
    # Common layouts: model.network, model.net, model.model
    candidates = [
        getattr(model, "network", None),
        getattr(model, "net", None),
        getattr(model, "model", None),
        model,
    ]
    for obj in candidates:
        if obj is None:
            continue
        # Common preprocess holders: preprocess, preprocess_layer, pre, pp
        for pre_name in ("preprocess", "preprocess_layer", "pre", "pp"):
            pre = getattr(obj, pre_name, None)
            if pre is not None and hasattr(pre, "nl"):
                return pre.nl
        # Sometimes NL is attached directly
        if hasattr(obj, "nl"):
            return obj.nl

    raise AttributeError(
        "Could not find neighbor-list builder on model. "
        "Expected something like model.network.preprocess.nl"
    )

def _get_potential_loss_config(params: dict) -> PotentialLossConfig:
    """Parse potential_model training knobs from params dict with safe defaults."""
    mp = params.get("model", {}).get("params", {}) or {}
    return PotentialLossConfig(
        e_unit=float(mp.get("e_unit", 1.0)),
        e_scale=float(mp.get("e_scale", 1.0)),
        e_loss_multiplier=float(mp.get("e_loss_multiplier", 1.0)),
        f_loss_multiplier=float(mp.get("f_loss_multiplier", 1.0)),
        use_force=bool(mp.get("use_force", True)),
        use_e_per_atom=bool(mp.get("use_e_per_atom", False)),
        log_e_per_atom=bool(mp.get("log_e_per_atom", False)),
    )


def _make_sparse_batch(
    batch: Dict[str, np.ndarray],
    *,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Convert a dense (B,N,...) batch to PiNN sparse dict tensors.

    Returns a dict with:
      coord: (B*N,3) requires_grad=True
      elems: (B*N,)
      ind_1: (B*N,1) structure ids
      _E_true: (B,)
      _F_true: (B*N,3)
    """
    R = torch.tensor(batch["coord"], dtype=torch.float32, device=device)  # (B,N,3)
    Z = torch.tensor(batch["elems"], dtype=torch.long, device=device)     # (B,N)
    E_true = torch.tensor(batch["e_data"], dtype=torch.float32, device=device)  # (B,)
    F_true = torch.tensor(batch["f_data"], dtype=torch.float32, device=device)  # (B,N,3)


    B, N, _ = R.shape

    coord = R.reshape(B * N, 3).detach().clone().requires_grad_(True)
    elems = Z.reshape(B * N)

    ind_1 = torch.zeros((B * N, 1), dtype=torch.long, device=device)
    ind_1[:, 0] = torch.arange(B, device=device).repeat_interleave(N)

    return {
        "coord": coord,
        "elems": elems,
        "ind_1": ind_1,
        "_E_true": E_true,
        "_F_true": F_true.reshape(B * N, 3),
    }


def iter_batches(
    data: Dict[str, np.ndarray],
    batch_size: int,
    *,
    shuffle: bool,
    seed: int = 0,
    repeat: bool = True,
) -> Iterator[Dict[str, np.ndarray]]:
    """Yield minibatches from the LJ dataset dict.

    This is a torch-friendly replacement for the TF input pipeline used in
    tests/test_potential.py. It yields numpy arrays; the training loop will
    convert them to torch tensors.

    Parameters
    ----------
    data
        Dict with keys: 'coord', 'elems', 'e_data', 'f_data'.
    batch_size
        Number of configurations per batch.
    shuffle
        Whether to shuffle indices each epoch.
    seed
        RNG seed for deterministic shuffling.
    repeat
        If True, loop forever; if False, yield one epoch only.

    Yields
    ------
    batch : dict
        Same keys as `data`, but batched on the first axis.
    """
    n = data["coord"].shape[0]
    rng = np.random.default_rng(seed)

    while True:
        idx = np.arange(n)
        if shuffle:
            rng.shuffle(idx)

        for start in range(0, n, batch_size):
            sel = idx[start:start + batch_size]
            if sel.size == 0:
                continue
            yield {
                "coord": data["coord"][sel],
                "elems": data["elems"][sel],
                "e_data": data["e_data"][sel],
                "f_data": data["f_data"][sel],
            }

        if not repeat:
            break


def _repeat_batches_from_built_examples(built_examples, *, opts, nl_builder):
    """
    Repeat forever over an already-built (and ideally cached/materialized) example iterable.

    This avoids rebuilding datasets each epoch and makes RAM cache effective.
    """
    while True:
        for batch in _iter_batches_from_examples(built_examples, opts=opts, nl_builder=nl_builder):
            yield batch


def train_and_evaluate(
    *,
    model,
    params,
    data,
    max_steps: int,
    eval_steps: int,
    batch_size_train: int,
    batch_size_eval: int,
    shuffle_buffer: int = 0,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "auto",
    **kwargs,
):
    """Train and evaluate a torch PiNN model on the LJ toy dataset.

    This implements the torch backend counterpart of the TF estimator training
    used in tests/test_potential.py. It consumes the raw `data` dict produced
    by `_get_lj_data()`.

    Metric convention (must match the test)
    --------------------------------------
    The test checks:

        results["METRICS/E_RMSE"] / e_scale  ≈  RMSE( E_pred/e_unit - e_data )
        results["METRICS/F_RMSE"] / e_scale  ≈  RMSE( F_pred/e_unit - f_data )

    Therefore we return:

        METRICS/E_RMSE = e_scale * RMSE( E_pred/e_unit - e_data )
        METRICS/F_RMSE = e_scale * RMSE( F_pred/e_unit - f_data )
        
        If use_force is False, METRICS/F_RMSE is reported as 0.0 (Torch behavior).

    Parameters
    ----------
    model
        Torch model is called as model(tensors_dict) where tensors follow PiNN sparse conventions.
    params : dict
        Model spec dict containing model_dir and model.params (e_unit, e_scale, e_dress).
    data : dict
        LJ dataset dict with keys 'coord', 'elems', 'e_data', 'f_data'.
    max_steps : int
        Training steps.
    eval_steps : int
        Number of eval batches to average metrics over.
    batch_size_train : int
        Training batch size.
    batch_size_eval : int
        Eval batch size.
    lr : float Default learning rate used only if params['optimizer'] is absent.
    seed : int
        RNG seed.
    device : str
        Torch device string.

    Returns
    -------
    dict
        Metrics dict with keys 'METRICS/E_RMSE' and 'METRICS/F_RMSE'.
    """

    # Torch input sources:
    # - existing tests: data is a numpy dict
    # - CLI path: train_yml/eval_yml point to dataset YAML
    train_yml = kwargs.pop("train_yml", None)
    eval_yml = kwargs.pop("eval_yml", None)

    cache = bool(kwargs.pop("cache", True))
    cache_ram = bool(kwargs.pop("cache_ram", True))
    preprocess_flag = bool(kwargs.pop("preprocess", True))
    scratch_dir = kwargs.pop("scratch_dir", None)
    log_every = int(kwargs.pop("log_every", 100))
    ckpt_every = int(kwargs.pop("ckpt_every", 1000))
    keep_ckpts = int(kwargs.pop("keep_ckpts", 5))
    resume = bool(kwargs.pop("resume", True))
    eval_every = int(kwargs.pop("eval_every", 1000))   # how often to run eval during training
    eval_batches = int(kwargs.pop("eval_batches", eval_steps))  # optional: cheaper periodic eval
    dev = resolve_device(device)
    device = str(dev)

    cfg = _get_potential_loss_config(params)
    e_unit = cfg.e_unit
    e_scale = cfg.e_scale

    model_dir = params.get("model_dir", None)
    if model_dir is None:
        raise ValueError("params['model_dir'] is required for torch training.")
    os.makedirs(model_dir, exist_ok=True)


    writer = SummaryWriter(log_dir=model_dir)  # or model_dir/"logs"
    try: 

        # Move model to device (once you have a real torch model)
        if hasattr(model, "to"):
            model = model.to(device)

        # Optimizer/scheduler/clipping from params.yml-style optimizer spec (generic for all torch nets)
        optim, scheduler, clip = build_optimizer_from_params(model, params, default_lr=lr)

        # Resume from checkpoint 

        start_step = 0
        if resume:
            start_step = _try_resume_from_checkpoint(
                model=model,
                optimizer=optim,
                scheduler=scheduler,
                model_dir=model_dir,
                device=device,
            )
            if start_step > 0:
                print(f"[torch] Resumed from checkpoint at step {start_step}")

        # ---- Build data iterators (numpy dict path OR YAML/CLI path) ----
        net_params = params.get("network", {}).get("params", {}) or {}
        atom_types = net_params.get("atom_types", None)
        rc = float(net_params.get("rc", 0.0))

        if train_yml is not None or eval_yml is not None:
            if train_yml is None or eval_yml is None:
                raise ValueError("Torch runtime requires both train_yml and eval_yml when using YAML datasets.")
            if atom_types is None:
                raise ValueError("params['network']['params']['atom_types'] is required for Torch YAML pipeline.")
            if rc <= 0.0:
                raise ValueError("params['network']['params']['rc'] must be > 0 for Torch YAML pipeline.")

            nl_builder = _get_nl_builder_from_model(model)

            # Cache directory: prefer explicit kwarg, fall back to model_dir
            if scratch_dir is None:
                scratch_dir = model_dir  # safe default

            train_opts = TorchDataOptions(
                batch_size=int(batch_size_train),
                shuffle_buffer=int(shuffle_buffer),
                atom_types=list(atom_types),
                rc=float(rc),
                scratch_dir=scratch_dir,
                cache=cache,
                cache_ram=cache_ram,
                device=device,
                preprocess=preprocess_flag,
            )
            eval_opts = TorchDataOptions(
                batch_size=int(batch_size_eval),
                shuffle_buffer=0,
                atom_types=list(atom_types),
                rc=float(rc),
                scratch_dir=scratch_dir,
                cache=cache,
                cache_ram=cache_ram,
                device=device,
                preprocess=preprocess_flag,
            )

            # ---- Build TRAIN once, then repeat batches forever ----
            built_train = build_dataset(
                train_yml,
                options=BuildOptions(
                    backend="torch",
                    cache=train_opts.cache,
                    scratch_dir=train_opts.scratch_dir,
                    cache_ram=train_opts.cache_ram,
                ),
                dataset_role="train",
            )
            # IMPORTANT: make train examples re-iterable + make RAM cache truly effective
            # Materialize only if explicitly requested (small datasets / tests).
            materialize = bool(kwargs.pop("materialize_dataset", False))
            if materialize:
                built_train = list(built_train)

            train_it = _repeat_batches_from_built_examples(
                built_train, opts=train_opts, nl_builder=nl_builder
            )

            use_yml_pipeline = True

            # ---- Build EVAL once and materialize (re-iterable across periodic evals) ----
            built_eval = build_dataset(
                eval_yml,
                options=BuildOptions(
                    backend="torch",
                    cache=eval_opts.cache,
                    scratch_dir=eval_opts.scratch_dir,
                    cache_ram=eval_opts.cache_ram,
                ),
                dataset_role="eval",
            )
            built_eval = list(built_eval)
            

        else:
            # Existing unit-test path: LJ toy numpy dict
            train_it = iter_batches(data, batch_size_train, shuffle=True, seed=seed, repeat=True)
            eval_it = iter_batches(data, batch_size_eval, shuffle=False, seed=seed, repeat=True)
            use_yml_pipeline = False
        
        # ---- atomic dressing (Torch parity with TF) ----
        model_params = params.get("model", {}).get("params", {}) or {}
        if params.get("model", {}).get("name") == "potential_model" and "e_dress" not in model_params:
            # init_params is TF-only: always feed it a tf.data.Dataset
            try:
                if use_yml_pipeline:
                    from pinn.io import load_tfrecord  # -> tf.data.Dataset
                    tf_train_ds = load_tfrecord(train_yml)
                else:
                    from pinn.io import load_numpy     # -> tf.data.Dataset
                    tf_train_ds = load_numpy(data)

                init_params(params, tf_train_ds)  # TF contract satisfied
                _inject_e_dress_into_torch_model(model, params)
                print("Torch model e_dress active:", getattr(model, "e_dress", None))

            except Exception as e:
                raise RuntimeError(
                    "Failed to initialize e_dress via TF init_params(). "
                    "This requires TensorFlow and a valid TF dataset input."
                ) from e

            # Persist params.yml
            params_path = os.path.join(model_dir, "params.yml")
            with open(params_path, "w") as f:
                yaml.safe_dump(params, f)
        
        def _run_eval(n_batches: int) -> tuple[float, float]:
            if hasattr(model, "eval"):
                model.eval()

            # --- TF semantics: TRAIN/EVAL happens in scaled label space ---
            # E_true_scaled = (E_true_raw - E_dress_struct) * e_scale
            # F_true_scaled = F_true_raw * e_scale
            model_params = (params.get("model", {}).get("params", {}) or {})
            e_scale = float(model_params.get("e_scale", 1.0))

            e_sq_sum = 0.0
            e_count = 0
            f_sq_sum = 0.0
            f_count = 0

            if use_yml_pipeline:
                eval_iter = _iter_batches_from_examples(built_eval, opts=eval_opts, nl_builder=nl_builder)
            else:
                eval_iter = iter_batches(
                    data,
                    batch_size_eval,
                    shuffle=False,
                    seed=seed,
                    repeat=False,
                )

            for _ in range(int(n_batches)):
                try:
                    batch = next(eval_iter)
                except StopIteration:
                    break

                if use_yml_pipeline:
                    tensors = {k: v for k, v in batch.items() if k not in ("e_data", "f_data")}
                    dev_t = next(model.parameters()).device
                    tensors = move_pinn_batch_to_device(tensors, dev_t)

                    # Raw labels from dataset (Hartree and Hartree/Å for water RuNNer)
                    E_true_raw = batch["e_data"].to(dev_t, non_blocking=True)
                    F_true_raw = _flatten_forces(batch["f_data"]).to(dev_t, non_blocking=True)

                    coord = tensors["coord"]
                    if (not coord.requires_grad) or (not coord.is_leaf):
                        coord = coord.detach().clone().requires_grad_(True)
                        tensors["coord"] = coord
                else:
                    sb = _make_sparse_batch(batch, device=device)
                    tensors = {k: v for k, v in sb.items() if not k.startswith("_")}

                    E_true_raw = sb["_E_true"]
                    F_true_raw = _flatten_forces(sb["_F_true"])

                    coord = tensors["coord"]
                    if (not coord.requires_grad) or (not coord.is_leaf):
                        coord = coord.detach().clone().requires_grad_(True)
                        tensors["coord"] = coord

                # -------- Dress-subtract energy labels (ONLY energy; dressing has zero force) --------
                if hasattr(model, "dress_struct"):
                    # returns per-structure dress in RAW label units (Hartree)
                    dress = model.dress_struct(tensors)
                    # Safety: ensure shape (B,)
                    if dress.ndim == 2 and dress.shape[1] == 1:
                        dress = dress[:, 0]
                else:
                    dress = 0.0

                # -------- Scale labels into TRAIN/EVAL label space (TF semantics) --------
                E_true_s = (E_true_raw - dress) * e_scale
                F_true_s = F_true_raw * e_scale

                # -------- Predict in TRAIN/EVAL label space --------
                if hasattr(model, "forward_train"):
                    E_pred_s = model.forward_train(tensors)
                else:
                    # Fallback: assume model(tensors) already returns TRAIN/EVAL energy (scaled-space)
                    E_pred_s = model(tensors)

                if E_pred_s.ndim == 2 and E_pred_s.shape[1] == 1:
                    E_pred_s = E_pred_s[:, 0]
                elif E_pred_s.ndim != 1:
                    raise ValueError(f"Expected E_pred shape (B,) or (B,1), got {tuple(E_pred_s.shape)}")

                # -------- Energy RMSE in scaled label space --------
                if cfg.log_e_per_atom:
                    ind_1 = tensors.get("ind_1", None)
                    if ind_1 is None:
                        counts = E_true_s.new_tensor([float(tensors["coord"].shape[0])]).clamp_min(1.0)
                    else:
                        B = int(E_true_s.shape[0])
                        counts = (
                            torch.bincount(ind_1[:, 0], minlength=B)
                            .to(E_true_s.dtype)
                            .clamp_min(1.0)
                        )
                    e_err = (E_pred_s / counts) - (E_true_s / counts)
                else:
                    e_err = E_pred_s - E_true_s

                e_sq_sum += float((e_err ** 2).sum().item())
                e_count += int(e_err.numel())

                # -------- Force RMSE in scaled label space --------
                if cfg.use_force:
                    # Forces must be gradients of TRAIN/EVAL energy (scaled-space)
                    dE_dR = torch.autograd.grad(E_pred_s.sum(), coord, create_graph=False)[0]
                    F_pred_s = -dE_dR
                    f_err = F_pred_s - F_true_s
                    f_sq_sum += float((f_err ** 2).sum().item())
                    f_count += int(f_err.numel())

            e_rmse = float(np.sqrt(e_sq_sum / max(e_count, 1)))
            f_rmse = float(np.sqrt(f_sq_sum / max(f_count, 1))) if cfg.use_force else 0.0
            return e_rmse, f_rmse
         

        # ---- training loop ----
        if hasattr(model, "train"):
            model.train()

        for step in range(start_step, int(max_steps)):
            batch = next(train_it)

            if use_yml_pipeline:
                # batch is already a torch sparse+preprocessed dict
                tensors = {k: v for k, v in batch.items() if k not in ("e_data", "f_data")}
                if "e_data" not in batch or "f_data" not in batch:
                    raise KeyError("YAML Torch pipeline batch must include 'e_data' and 'f_data' for potential_model tests.")
                
                dev_t = next(model.parameters()).device
                tensors = move_pinn_batch_to_device(tensors, dev_t)
                
                E_true = batch["e_data"].to(dev_t, non_blocking=True)
                F_true = _flatten_forces(batch["f_data"]).to(dev_t, non_blocking=True)
                # enforce invariant for force autograd
                coord = tensors["coord"]
                if not coord.requires_grad:
                    coord.requires_grad_(True)
            else:
                # LJ numpy dict path
                sb = _make_sparse_batch(batch, device=device)
                tensors = {k: v for k, v in sb.items() if not k.startswith("_")}
                E_true = sb["_E_true"]
                F_true = sb["_F_true"]
                F_true = _flatten_forces(F_true)
                coord = tensors["coord"]
                if not coord.requires_grad or not coord.is_leaf:
                    coord = coord.detach().clone().requires_grad_(True)
                    tensors["coord"] = coord

            # ---------------------------
            # TRAIN in scaled label space (TF semantics)
            # ---------------------------

            # 0) fetch e_scale (already read above as cfg.e_scale, but keep explicit)
            e_scale = cfg.e_scale

            # 1) dress-subtract energy labels (dress has no force contribution)
            if hasattr(model, "dress_struct"):
                dress = model.dress_struct(tensors)
                if torch.is_tensor(dress) and dress.ndim == 2 and dress.shape[1] == 1:
                    dress = dress[:, 0]
            else:
                dress = 0.0

            E_true_s = (E_true - dress) * e_scale
            F_true_s = F_true * e_scale

            # 2) forward TRAIN energy (scaled-space)
            if hasattr(model, "forward_train"):
                E_pred_s = model.forward_train(tensors)
            else:
                # If your model doesn't implement this, you *must* add it (or wrap it)
                raise TypeError("Torch potential_model must implement forward_train() for TF-parity training.")

            if E_pred_s.ndim == 2 and E_pred_s.shape[1] == 1:
                E_pred_s = E_pred_s[:, 0]
            elif E_pred_s.ndim != 1:
                raise ValueError(f"Expected E_pred_s shape (B,) or (B,1), got {tuple(E_pred_s.shape)}")

            # 3) forces from TRAIN energy
            if cfg.use_force:
                dE_dR = torch.autograd.grad(
                    E_pred_s.sum(),
                    coord,
                    create_graph=True,   # need param grads through forces
                    retain_graph=True,
                )[0]
                F_pred_s = -dE_dR
            else:
                F_pred_s = None  # type: ignore[assignment]

            # 4) losses in TRAIN space (no /e_unit here)
            if cfg.use_e_per_atom:
                B = int(E_true_s.shape[0])
                counts = torch.bincount(tensors["ind_1"][:, 0], minlength=B).to(E_true_s.dtype).clamp_min(1.0)
                e_err = (E_pred_s / counts) - (E_true_s / counts)
            else:
                e_err = E_pred_s - E_true_s
            e_loss = (e_err ** 2).mean()

            if cfg.use_force:
                assert F_pred_s.shape == F_true_s.shape, (F_pred_s.shape, F_true_s.shape)
                f_err = F_pred_s - F_true_s
                f_loss = (f_err ** 2).mean()
            else:
                f_loss = torch.zeros((), dtype=e_loss.dtype, device=e_loss.device)

            loss = cfg.e_loss_multiplier * e_loss + cfg.f_loss_multiplier * f_loss

            if optim is not None:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                apply_grad_clipping(model, clip)
                optim.step()
                if (step % log_every) == 0:
                    # ---- metric errors: must match eval convention ----
                    # metrics in TRAIN space (scaled)
                    if cfg.log_e_per_atom:
                        B = int(E_true_s.shape[0])
                        counts = (
                            torch.bincount(tensors["ind_1"][:, 0], minlength=B)
                            .to(E_true_s.dtype)
                            .clamp_min(1.0)
                        )
                        e_err_metric = (E_pred_s / counts) - (E_true_s / counts)
                    else:
                        e_err_metric = E_pred_s - E_true_s

                    train_e_rmse = torch.sqrt((e_err_metric ** 2).mean()).item()

                    if cfg.use_force:
                        f_err_metric = F_pred_s - F_true_s
                        train_f_rmse = torch.sqrt((f_err_metric ** 2).mean()).item()
                    else:
                        train_f_rmse = 0.0

                    lr_now = optim.param_groups[0]["lr"] if optim is not None else float("nan")
                    writer.add_scalar("train/loss", float(loss.item()), step)
                    writer.add_scalar("train/E_RMSE", train_e_rmse, step)
                    writer.add_scalar("train/F_RMSE", train_f_rmse, step)
                    writer.add_scalar("train/lr", float(lr_now), step)
                    writer.flush()
                if scheduler is not None:
                    scheduler.step()
                
                if ckpt_every > 0 and (step % ckpt_every) == 0 and step > start_step:
                    _save_checkpoint(
                        model=model,
                        optimizer=optim,
                        scheduler=scheduler,
                        params=params,
                        step=step,
                        model_dir=model_dir,
                        keep_last=keep_ckpts,
                    )
                
            # Periodic evaluation curve (held-out eval_yml / eval_it)
            if eval_every > 0 and step > 0 and (step % eval_every) == 0:
                e_rmse_now, f_rmse_now = _run_eval(eval_batches)
                writer.add_scalar("eval/E_RMSE", e_rmse_now, step)
                writer.add_scalar("eval/F_RMSE", f_rmse_now, step)
                writer.flush()

                if hasattr(model, "train"):
                    model.train()  # switch back for next step

        # ---- evaluation loop (compute RMSE) ----
        if hasattr(model, "eval"):
            model.eval()

        e_rmse, f_rmse = _run_eval(eval_steps)
        writer.add_scalar("eval/E_RMSE", e_rmse, max_steps)
        writer.add_scalar("eval/F_RMSE", f_rmse, max_steps)
        writer.flush()

        # Save checkpoint
        if hasattr(model, "state_dict"):
            to_save = model
            # If later you ever wrap with nn.DataParallel / DDP
            if hasattr(model, "module"):
                to_save = model.module

            ckpt = {
                "model_state_dict": to_save.state_dict(),
                "params": params,
                # Optional metadata (harmless, helps debugging)
                "pytorch_version": torch.__version__,
            }
            torch.save(ckpt, os.path.join(model_dir, "model.pt"))

        _save_checkpoint(
            model=model,
            optimizer=optim,
            scheduler=scheduler,
            params=params,
            step=max_steps,
            model_dir=model_dir,
            keep_last=keep_ckpts,
        )
    finally:
        writer.close()    

    return {
        "METRICS/E_RMSE": e_rmse,
        "METRICS/F_RMSE": f_rmse,
    }



