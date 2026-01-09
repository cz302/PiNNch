# pinn/torch/input_pipeline.py
"""
Torch input pipeline utilities for PiNN.

This module turns:
- a dataset YAML (train.yml/eval.yml), OR
- an in-memory numpy dict (existing tests)

into an iterator over *batched & preprocessed* torch tensors suitable for PiNet/PiNet2.

Responsibilities:
- build dataset spec via pinn.io.build_dataset (torch backend path)
- optional caching already handled in build_dataset
- sparse batching (concatenate atoms/edges across structures)
- preprocessing (ind_2/shift + diff/dist + prop) via preprocess_batch_torch

We avoid importing TensorFlow unless build_dataset needs it for TFRecord reading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from pinn.io.build_dataset import build_dataset, BuildOptions
from pinn.io.torch.preprocess import preprocess_batch_torch


@dataclass(frozen=True)
class TorchDataOptions:
    """
    Options controlling Torch data pipeline.

    Attributes:
        batch_size: number of structures per batch
        shuffle_buffer: if >0, do a simple reservoir shuffle on structures
        atom_types: list of atomic numbers for one-hot encoding
        rc: cutoff radius for neighbor list (must match network params)
        scratch_dir: cache directory (optional)
        cache: enable cache (npz/ram) via build_dataset
        cache_ram: enable RAM cache in addition to disk cache
        device: torch device string, e.g. "cpu" or "cuda"
    """
    batch_size: int
    shuffle_buffer: int
    atom_types: Sequence[int]
    rc: float
    scratch_dir: Optional[str] = None
    cache: bool = True
    cache_ram: bool = True
    device: str = "cpu"

    # Preprocessing knob
    preprocess: bool = True


def _to_torch(x: Any, device: torch.device) -> torch.Tensor:
    """Convert input to a torch.Tensor on `device` safely.

    This function is the NumPy→Torch boundary for the input pipeline.

    Why we do extra checks:
    - `torch.as_tensor(numpy_array)` may share memory with NumPy.
    - Arrays loaded from `.npz`/memmap/views can be non-writable.
    - If any downstream code performs in-place writes on a Tensor backed by
      a non-writable NumPy buffer, behavior is undefined (PyTorch warns).

    Policy:
    - If `x` is a torch.Tensor: move to device (no copy unless needed).
    - If `x` is a NumPy array (or memmap) and is not writeable: copy once.
    - Otherwise: use `torch.as_tensor` for zero-copy when safe.

    Args:
        x: torch.Tensor, numpy array/memmap, scalar, list, etc.
        device: Target torch device.

    Returns:
        torch.Tensor on `device`.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)

    # Avoid undefined behavior from non-writable NumPy buffers (common for np.load/.npz).
    if isinstance(x, np.ndarray) and not x.flags.writeable:
        x = np.array(x, copy=True)

    return torch.as_tensor(x, device=device)


def _example_to_structure(example: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Convert a single structure example (numpy dict) to torch tensors and add ind_1.

    Required keys:
      - coord: (N,3)
      - elems or z: (N,)
    Optional:
      - cell: (3,3) or None

    Returns:
      dict with torch tensors and ind_1 (N,1) all zeros (single structure).
    """
    out: Dict[str, Any] = {}
    coord = _to_torch(example["coord"], device).to(dtype=torch.float32)
    out["coord"] = coord

    if "elems" in example:
        elems = _to_torch(example["elems"], device).to(dtype=torch.long)
    else:
        elems = _to_torch(example["z"], device).to(dtype=torch.long)
    out["elems"] = elems

    n = int(coord.shape[0])
    out["ind_1"] = torch.zeros((n, 1), dtype=torch.long, device=device)

    if "cell" in example and example["cell"] is not None:
        cell = _to_torch(example["cell"], device).to(dtype=torch.float32)
        out["cell"] = cell

    # pass through labels if present
    for k in ("e_data", "energy", "E"):
        if k in example:
            out[k] = _to_torch(example[k], device).to(dtype=torch.float32)
            break
    for k in ("f_data", "forces", "F"):
        if k in example:
            out[k] = _to_torch(example[k], device).to(dtype=torch.float32)
            break

    return out


def _sparse_batch_structures(structs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Sparse-batch a list of per-structure dicts into a single batched dict.

    We only batch the essentials needed by preprocess_batch_torch + models:
      - coord, elems, ind_1
      - optional cell (stacked if present for all)
      - optional labels (E/forces)
    """
    if len(structs) == 0:
        raise ValueError("Cannot batch empty structure list.")

    device = structs[0]["coord"].device
    coords = []
    elems = []
    ind_1 = []
    cells = []
    have_cell = True
    e_list = []
    f_list = []
    have_e = True
    have_f = True

    atom_offset = 0
    for bi, s in enumerate(structs):
        c = s["coord"]
        z = s["elems"]
        n = int(c.shape[0])
        coords.append(c)
        elems.append(z)

        # ind_1[:,0] is structure id
        ind_1.append(torch.full((n, 1), bi, dtype=torch.long, device=device))

        if "cell" in s:
            cells.append(s["cell"])
        else:
            have_cell = False

        if have_e:
            if "e_data" in s:
                e_list.append(s["e_data"].reshape(-1))
            elif "energy" in s:
                e_list.append(s["energy"].reshape(-1))
            elif "E" in s:
                e_list.append(s["E"].reshape(-1))
            else:
                have_e = False

        if have_f:
            if "f_data" in s:
                f_list.append(s["f_data"])
            elif "forces" in s:
                f_list.append(s["forces"])
            elif "F" in s:
                f_list.append(s["F"])
            else:
                have_f = False

        atom_offset += n

    out: Dict[str, Any] = {
        "coord": torch.cat(coords, dim=0),
        "elems": torch.cat(elems, dim=0),
        "ind_1": torch.cat(ind_1, dim=0),
    }

    if have_cell and len(cells) == len(structs):
        out["cell"] = torch.stack(cells, dim=0)

    if have_e and e_list:
        out["e_data"] = torch.cat(e_list, dim=0)

    if have_f and f_list:
        out["f_data"] = torch.cat(f_list, dim=0)

    return out


def _iter_batches_from_examples(
    examples: Iterable[Mapping[str, Any]],
    *,
    opts: TorchDataOptions,
    nl_builder: Any = None,
) -> Iterator[Dict[str, Any]]:
    """
    Yield torch batches from an example stream.

    If opts.preprocess is True, batches are augmented with:
      - prop, ind_2, shift, diff, dist
    """
    device = torch.device(opts.device)

    # Simple structure-level shuffle buffer (optional).
    buf: list[Mapping[str, Any]] = []
    rng = np.random.default_rng(0)

    def pop_one() -> Optional[Mapping[str, Any]]:
        if not buf:
            return None
        if opts.shuffle_buffer > 0 and len(buf) > 1:
            j = int(rng.integers(0, len(buf)))
            return buf.pop(j)
        return buf.pop(0)

    batch_structs = []
    for ex in examples:
        buf.append(ex)

        item = pop_one()
        if item is None:
            continue

        s = _example_to_structure(item, device)
        batch_structs.append(s)

        if len(batch_structs) == opts.batch_size:
            batch = _sparse_batch_structures(batch_structs)

            coord = batch["coord"]
            if not coord.is_floating_point():
                batch["coord"] = coord.float()
                coord = batch["coord"]

            # IMPORTANT: in-place, keep identity
            if not coord.requires_grad:
                coord.requires_grad_(True)

            if opts.preprocess:
                if nl_builder is None:
                    raise ValueError("nl_builder must be provided when preprocess=True.")
                batch = preprocess_batch_torch(
                    batch,
                    atom_types=opts.atom_types,
                    rc=opts.rc,
                    nl_builder=nl_builder,
                    make_diff_dist=False,
                )

            yield batch
            batch_structs = []

    # flush remainder (useful for eval)
    if batch_structs:
        batch = _sparse_batch_structures(batch_structs)

        coord = batch["coord"]
        if not coord.is_floating_point():
            batch["coord"] = coord.float()
            coord = batch["coord"]

        # IMPORTANT: in-place, keep identity
        if not coord.requires_grad:
            coord.requires_grad_(True)

        if opts.preprocess:
            if nl_builder is None:
                raise ValueError("nl_builder must be provided when preprocess=True.")
            batch = preprocess_batch_torch(
                batch,
                atom_types=opts.atom_types,
                rc=opts.rc,
                nl_builder=nl_builder,
            )

        yield batch


def make_torch_dataloader_from_yml(
    yml_path: str,
    *,
    dataset_role: str,
    opts: TorchDataOptions,
    nl_builder: Any,
) -> Iterable[Dict[str, Any]]:
    """
    Build a Torch batch iterator from a dataset YAML file.

    Args:
        yml_path: train.yml / eval.yml
        dataset_role: "train" or "eval"
        opts: TorchDataOptions
        nl_builder: neighborlist builder from the network preprocess layer (CellListNLPyTorch)

    Returns:
        Iterable yielding torch batch dicts.
    """
    built = build_dataset(
        yml_path,
        options=BuildOptions(
            backend="torch",
            cache=opts.cache,
            scratch_dir=opts.scratch_dir,
            cache_ram=opts.cache_ram,
        ),
        dataset_role=dataset_role,
    )
    return _iter_batches_from_examples(built, opts=opts, nl_builder=nl_builder)
