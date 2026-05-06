"""
Simplified release-site weight modulation experiment.

This file intentionally avoids the broad all-in-one training machinery.  It
contains only:
  - a primary recurrent SNN state,
  - a release-site MLP,
  - a forward loop that periodically updates weights from recent activity.

Release sites are fixed positions in hidden/output neuron space.  The MLP sees
recent input spikes, hidden spikes, output membrane values, and compact local
summaries of the current weights around each release site.  It outputs release
amplitudes and spreads, which are converted into spatial fields and applied to
incoming weights.
"""

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_LOG_PATH = Path(__file__).with_name("snn_weight_release_mod_log.jsonl")

GridDim = Literal["1d", "2d"]
SpreadMode = Literal["uniform", "normal"]
UpdateMode = Literal["add", "sub"]


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"yes", "true", "t", "1", "y"}:
        return True
    if text in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def set_seed(seed: Optional[int]):
    if seed is None:
        return
    seed_int = int(seed)
    random.seed(seed_int)
    np.random.seed(seed_int)
    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_int)


def resolve_h5_path(cache_dir: str, cache_subdir: Optional[str], file_name: str) -> str:
    if os.path.isabs(file_name):
        return file_name
    base = Path(os.path.expanduser(str(cache_dir or ".")))
    if cache_subdir:
        base = base / str(cache_subdir)
    return str(base / file_name)


def load_h5(cache_dir: str, cache_subdir: Optional[str], train_file: str, test_file: str):
    train_path = resolve_h5_path(cache_dir, cache_subdir, train_file)
    test_path = resolve_h5_path(cache_dir, cache_subdir, test_file)
    train_h5 = h5py.File(train_path, "r")
    test_h5 = h5py.File(test_path, "r")
    return train_h5["spikes"], train_h5["labels"], test_h5["spikes"], test_h5["labels"]


def load_validation_split(cache_dir: str, cache_subdir: Optional[str], val_file: Optional[str]):
    if not val_file:
        return None, None
    val_path = resolve_h5_path(cache_dir, cache_subdir, val_file)
    val_h5 = h5py.File(val_path, "r")
    return val_h5["spikes"], val_h5["labels"]


def ensure_run_dir(base_dir: str) -> Path:
    base = Path(os.path.expanduser(str(base_dir)))
    base.mkdir(parents=True, exist_ok=True)
    idx = 1
    while True:
        run_dir = base / f"Run_{idx}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            idx += 1


def sparse_data_generator_from_hdf5_spikes(
    x_data,
    y_data,
    batch_size: int,
    nb_steps: int,
    nb_units: int,
    max_time: float,
    shuffle: bool = True,
    indices: Optional[np.ndarray] = None,
):
    labels_full = np.asarray(y_data, dtype=int)
    sample_index = np.arange(len(labels_full)) if indices is None else np.asarray(indices)
    if shuffle:
        np.random.shuffle(sample_index)
    number_of_batches = len(sample_index) // int(batch_size)
    firing_times, units_fired = x_data["times"], x_data["units"]
    time_bins = np.linspace(0.0, float(max_time), num=int(nb_steps), endpoint=False)

    for counter in range(number_of_batches):
        batch_index = sample_index[int(batch_size) * counter:int(batch_size) * (counter + 1)]
        coo = [[], [], []]
        for bc, idx in enumerate(batch_index):
            times = np.asarray(firing_times[idx])
            units = np.asarray(units_fired[idx], dtype=np.int64)
            valid = (units >= 0) & (units < int(nb_units)) & (times >= 0.0) & (times <= float(max_time))
            times = times[valid]
            units = units[valid]
            bins = np.searchsorted(time_bins, times, side="right") - 1
            bins = np.clip(bins, 0, int(nb_steps) - 1)
            coo[0].extend([bc] * len(bins))
            coo[1].extend(bins.tolist())
            coo[2].extend(units.tolist())

        i = torch.LongTensor(coo).to(device)
        v = torch.ones(len(coo[0]), device=device, dtype=dtype)
        x_batch = torch.sparse_coo_tensor(
            i,
            v,
            torch.Size([len(batch_index), int(nb_steps), int(nb_units)]),
            dtype=dtype,
            device=device,
        ).coalesce()
        y_batch = torch.tensor(labels_full[batch_index], device=device, dtype=torch.long)
        yield x_batch, y_batch


def stratified_split_indices(y_data, val_fraction: float = 0.1, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    y_np = np.asarray(y_data, dtype=int)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(y_np):
        cls_idx = np.where(y_np == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(len(cls_idx) * float(val_fraction)))
        val_idx.extend(cls_idx[:n_val])
        train_idx.extend(cls_idx[n_val:])
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def maybe_limit_indices(indices: np.ndarray, limit: Optional[int], seed: int) -> np.ndarray:
    if limit is None or int(limit) <= 0 or len(indices) <= int(limit):
        return indices
    rng = np.random.default_rng(seed)
    chosen = np.asarray(indices).copy()
    rng.shuffle(chosen)
    return np.sort(chosen[:int(limit)])


def save_indices(path: Path, train_idx: np.ndarray, val_idx: Optional[np.ndarray], meta: Optional[Dict[str, Any]] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_idx=train_idx,
        val_idx=val_idx if val_idx is not None else np.asarray([], dtype=np.int64),
        meta=meta or {},
    )


def load_indices(path: Union[str, Path]) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    data = np.load(Path(path).expanduser(), allow_pickle=True)
    train_idx = data["train_idx"]
    val_idx_arr = data["val_idx"]
    val_idx = val_idx_arr if val_idx_arr.size > 0 else None
    meta = dict(data["meta"].item()) if "meta" in data else {}
    return train_idx, val_idx, meta


def normalize_optional_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return text


def collect_stockpile_checkpoints(stockpile_dir: Union[str, Path]) -> List[Path]:
    root = Path(os.path.expanduser(str(stockpile_dir))).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Stockpile directory not found: {root}")
    run_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("Run_")]
    if not run_dirs:
        run_dirs = [p for p in root.glob("**/Run_*") if p.is_dir()]
    ckpts: List[Path] = []
    for run_dir in run_dirs:
        fold_dir = run_dir / "Fold_1"
        candidates = [
            fold_dir / "chosen.pth",
            fold_dir / "chosen",
            fold_dir / "snn_best.pth",
            fold_dir / "best_snn.pth",
        ]
        ckpt = next((candidate for candidate in candidates if candidate.exists()), None)
        if ckpt is not None:
            ckpts.append(ckpt)
    return ckpts


def pick_random_stockpile_ckpt(stockpile_dir: Union[str, Path], seed: Optional[int] = None) -> Path:
    ckpts = collect_stockpile_checkpoints(stockpile_dir)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {stockpile_dir}")
    rng = random.Random(int(seed)) if seed is not None else random.Random()
    chosen = rng.choice(ckpts)
    print(f"[info] Using stockpile base SNN: {chosen}")
    return chosen


class SurrGradSpike(torch.autograd.Function):
    scale = 100.0

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input / (SurrGradSpike.scale * torch.abs(inp) + 1.0) ** 2


spike_fn = SurrGradSpike.apply


def nearest_rect_grid(n: int) -> Tuple[int, int]:
    n = max(1, int(n))
    rows = max(1, int(math.floor(math.sqrt(n))))
    cols = max(1, int(math.ceil(n / rows)))
    while rows * cols < n:
        rows += 1
        cols = max(1, int(math.ceil(n / rows)))
    return rows, cols


def neuron_coords(n: int, grid_dim: GridDim, dev=None) -> torch.Tensor:
    dev = dev or device
    if grid_dim == "1d":
        return torch.arange(n, device=dev, dtype=dtype).unsqueeze(1)
    rows, cols = nearest_rect_grid(n)
    idx = torch.arange(n, device=dev)
    r = torch.div(idx, cols, rounding_mode="floor").to(dtype)
    c = (idx % cols).to(dtype)
    return torch.stack([r, c], dim=1)


def release_site_indices(n: int, count: int, grid_dim: GridDim, dev=None) -> torch.Tensor:
    dev = dev or device
    count = max(0, int(count))
    if count <= 0:
        return torch.empty((0,), device=dev, dtype=torch.long)
    if grid_dim == "1d":
        if count == 1:
            return torch.tensor([n // 2], device=dev, dtype=torch.long)
        pos = torch.linspace(0, n - 1, count, device=dev)
        return torch.round(pos).long().clamp(0, n - 1)

    rows, cols = nearest_rect_grid(n)
    r_count = max(1, int(round(math.sqrt(count))))
    c_count = max(1, int(math.ceil(count / r_count)))
    rr = torch.linspace(0, rows - 1, r_count, device=dev)
    cc = torch.linspace(0, cols - 1, c_count, device=dev)
    sites: List[int] = []
    for r in rr:
        for c in cc:
            idx = int(round(float(r))) * cols + int(round(float(c)))
            if idx < n:
                sites.append(idx)
            if len(sites) >= count:
                break
        if len(sites) >= count:
            break
    if not sites:
        sites = [min(n - 1, (rows // 2) * cols + (cols // 2))]
    return torch.tensor(sites[:count], device=dev, dtype=torch.long)


def pairwise_dist_to_sites(n: int, sites: torch.Tensor, grid_dim: GridDim) -> torch.Tensor:
    coords = neuron_coords(n, grid_dim, dev=sites.device)
    site_coords = coords.index_select(0, sites)
    diff = coords.unsqueeze(0) - site_coords.unsqueeze(1)
    return torch.linalg.norm(diff, dim=2)


def max_grid_distance(n: int, grid_dim: GridDim, dev=None) -> torch.Tensor:
    dev = dev or device
    if grid_dim == "1d":
        return torch.tensor(float(max(1, n - 1)), device=dev, dtype=dtype)
    rows, cols = nearest_rect_grid(n)
    return torch.tensor(math.sqrt(max(1, rows - 1) ** 2 + max(1, cols - 1) ** 2), device=dev, dtype=dtype)


def fixed_summary_kernels(
    n: int,
    sites: torch.Tensor,
    grid_dim: GridDim,
    width: float,
    mode: SpreadMode = "uniform",
) -> torch.Tensor:
    """Release-site neighborhoods used to summarize current weights."""
    if sites.numel() == 0:
        return torch.empty((0, n), device=sites.device, dtype=dtype)
    dist = pairwise_dist_to_sites(n, sites, grid_dim)
    width = max(float(width), 0.0)
    if mode == "normal":
        sigma = max(width, 1e-6)
        weights = torch.exp(-0.5 * (dist / sigma) ** 2)
    else:
        if width <= 0:
            weights = (dist <= 0.5).to(dtype)
        else:
            weights = (dist <= width).to(dtype)
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)


def spread_kernels(
    n: int,
    sites: torch.Tensor,
    spread01: torch.Tensor,
    grid_dim: GridDim,
    mode: SpreadMode,
) -> torch.Tensor:
    """
    Dynamic release kernels.

    spread01 is [B,R].  0 means effectively only the release-site neuron; 1
    means the whole axis/grid can receive effect.  Overlaps are normalized later
    when release amplitudes are converted into a neuron field.
    """
    if sites.numel() == 0:
        return torch.empty((spread01.size(0), 0, n), device=spread01.device, dtype=spread01.dtype)
    dist = pairwise_dist_to_sites(n, sites, grid_dim).to(device=spread01.device, dtype=spread01.dtype)
    max_dist = max_grid_distance(n, grid_dim, dev=spread01.device).to(dtype=spread01.dtype)
    spread = torch.clamp(spread01, 0.0, 1.0).unsqueeze(2)
    if mode == "normal":
        sigma = (spread * max_dist).clamp_min(1e-6)
        weights = torch.exp(-0.5 * (dist.unsqueeze(0) / sigma) ** 2)
        point = (dist.unsqueeze(0) <= 0.5).to(weights.dtype)
        weights = torch.where(spread <= 1e-6, point, weights)
        return weights
    radius = spread * max_dist
    return (dist.unsqueeze(0) <= radius.clamp_min(0.5)).to(spread01.dtype)


def incoming_weight_stats(matrix: torch.Tensor, target_kernels: torch.Tensor) -> torch.Tensor:
    """
    Summarize incoming weights around release sites.

    matrix is [B, source, target].
    target_kernels is [R, target], one normalized local neighborhood per site.

    For each release site we compute three stats over incoming weights into the
    site's local target-neuron neighborhood:
      mean weight, mean absolute weight, and standard deviation.
    """
    if target_kernels.numel() == 0:
        return matrix.new_zeros((matrix.size(0), 0))
    kernels = target_kernels.to(device=matrix.device, dtype=matrix.dtype)
    denom = float(matrix.size(1))
    mean = torch.einsum("bst,rt->br", matrix, kernels) / max(1.0, denom)
    abs_mean = torch.einsum("bst,rt->br", matrix.abs(), kernels) / max(1.0, denom)
    second = torch.einsum("bst,rt->br", matrix * matrix, kernels) / max(1.0, denom)
    std = torch.sqrt(torch.clamp(second - mean * mean, min=0.0))
    return torch.cat([mean, abs_mean, std], dim=1)


def tensor_summary(tensor: torch.Tensor, include_shape: bool = True) -> Dict[str, Any]:
    data = tensor.detach().float()
    out: Dict[str, Any] = {}
    if include_shape:
        out["shape"] = list(data.shape)
    if data.numel() == 0:
        out.update({
            "mean": 0.0,
            "abs_mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "finite": True,
        })
        return out
    finite = torch.isfinite(data).all()
    out.update({
        "mean": float(data.mean().item()),
        "abs_mean": float(data.abs().mean().item()),
        "std": float(data.std(unbiased=False).item()),
        "min": float(data.min().item()),
        "max": float(data.max().item()),
        "finite": bool(finite.item()),
    })
    return out


def summarize_modulation_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not events:
        return {"count": 0}

    def avg_nested_abs_mean(key: str) -> float:
        return float(sum(event[key]["abs_mean"] for event in events) / len(events))

    def avg_nested_mean(key: str) -> float:
        return float(sum(event[key]["mean"] for event in events) / len(events))

    return {
        "count": len(events),
        "first_step": int(events[0]["step"]),
        "last_step": int(events[-1]["step"]),
        "avg_amp_w1_abs_mean": avg_nested_abs_mean("amp_w1"),
        "avg_amp_v1_abs_mean": avg_nested_abs_mean("amp_v1"),
        "avg_amp_w2_abs_mean": avg_nested_abs_mean("amp_w2"),
        "avg_spread_h_mean": avg_nested_mean("spread_h"),
        "avg_spread_o_mean": avg_nested_mean("spread_o"),
        "avg_delta_w1_abs_mean": avg_nested_abs_mean("delta_w1"),
        "avg_delta_v1_abs_mean": avg_nested_abs_mean("delta_v1"),
        "avg_delta_w2_abs_mean": avg_nested_abs_mean("delta_w2"),
    }


def write_run_log(log_path: Optional[str], payload: Dict[str, Any]) -> Path:
    path = Path(log_path).expanduser() if log_path else DEFAULT_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
    return path


@dataclass
class ReleaseConfig:
    nb_inputs: int = 700
    nb_hidden: int = 256
    nb_outputs: int = 20
    nb_steps: int = 100
    time_step: float = 1e-3
    tau_syn: float = 10e-3
    tau_mem: float = 20e-3
    ann_interval: int = 3
    release_hidden: int = 64
    release_output: int = 8
    grid_dim: GridDim = "2d"
    spread_mode: SpreadMode = "normal"
    summary_width: float = 2.0
    summary_mode: SpreadMode = "uniform"
    update_mode: UpdateMode = "add"
    delta_scale: float = 0.05
    sub_scale: float = 5.0
    hidden_sizes: Tuple[int, ...] = (256,)


class ReleaseSiteMLP(nn.Module):
    """
    MLP that emits release amplitudes and spreads.

    Hidden release sites affect incoming w1 and v1 columns.  Output release
    sites affect incoming w2 columns.
    """

    def __init__(self, cfg: ReleaseConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden_sites = release_site_indices(cfg.nb_hidden, cfg.release_hidden, cfg.grid_dim)
        self.output_sites = release_site_indices(cfg.nb_outputs, cfg.release_output, cfg.grid_dim)
        self.hidden_summary = fixed_summary_kernels(
            cfg.nb_hidden, self.hidden_sites, cfg.grid_dim, cfg.summary_width, cfg.summary_mode
        )
        self.output_summary = fixed_summary_kernels(
            cfg.nb_outputs, self.output_sites, cfg.grid_dim, cfg.summary_width, cfg.summary_mode
        )
        self.hidden_release_count = int(self.hidden_sites.numel())
        self.output_release_count = int(self.output_sites.numel())

        activity_dim = cfg.nb_inputs + cfg.nb_hidden + cfg.nb_outputs
        summary_dim = 3 * (2 * self.hidden_release_count + self.output_release_count)
        self.input_dim = activity_dim + summary_dim
        self.output_dim = 3 * self.hidden_release_count + 2 * self.output_release_count

        layers: List[nn.Module] = []
        prev = self.input_dim
        for size in cfg.hidden_sizes:
            layers.append(nn.Linear(prev, int(size)))
            layers.append(nn.ReLU())
            prev = int(size)
        layers.append(nn.Linear(prev, self.output_dim))
        self.net = nn.Sequential(*layers)

        with torch.no_grad():
            for mod in self.net:
                if isinstance(mod, nn.Linear):
                    nn.init.normal_(mod.weight, mean=0.0, std=1e-4)
                    nn.init.zeros_(mod.bias)

    def build_input(
        self,
        in_flat: torch.Tensor,
        hid_flat: torch.Tensor,
        out_flat: torch.Tensor,
        w1_local: torch.Tensor,
        v1_local: torch.Tensor,
        w2_local: torch.Tensor,
    ) -> torch.Tensor:
        h_kernel = self.hidden_summary.to(device=w1_local.device, dtype=w1_local.dtype)
        o_kernel = self.output_summary.to(device=w2_local.device, dtype=w2_local.dtype)
        summaries = [
            incoming_weight_stats(w1_local, h_kernel),
            incoming_weight_stats(v1_local, h_kernel),
            incoming_weight_stats(w2_local, o_kernel),
        ]
        return torch.cat([in_flat, hid_flat, out_flat] + summaries, dim=1)

    def forward(self, mlp_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.net(mlp_input)
        rh = self.hidden_release_count
        ro = self.output_release_count
        offset = 0
        amp_w1 = torch.tanh(raw[:, offset:offset + rh]); offset += rh
        amp_v1 = torch.tanh(raw[:, offset:offset + rh]); offset += rh
        spread_h = torch.sigmoid(raw[:, offset:offset + rh]); offset += rh
        amp_w2 = torch.tanh(raw[:, offset:offset + ro]); offset += ro
        spread_o = torch.sigmoid(raw[:, offset:offset + ro])
        return {
            "amp_w1": amp_w1,
            "amp_v1": amp_v1,
            "spread_h": spread_h,
            "amp_w2": amp_w2,
            "spread_o": spread_o,
        }


def setup_primary_state(cfg: ReleaseConfig) -> Dict[str, torch.Tensor]:
    alpha_h = math.exp(-cfg.time_step / cfg.tau_syn)
    beta_h = math.exp(-cfg.time_step / cfg.tau_mem)
    alpha_o = alpha_h
    beta_o = beta_h
    w1 = torch.empty((cfg.nb_inputs, cfg.nb_hidden), device=device, dtype=dtype)
    v1 = torch.empty((cfg.nb_hidden, cfg.nb_hidden), device=device, dtype=dtype)
    w2 = torch.empty((cfg.nb_hidden, cfg.nb_outputs), device=device, dtype=dtype)
    nn.init.normal_(w1, mean=0.0, std=0.02)
    nn.init.normal_(v1, mean=0.0, std=0.02)
    nn.init.normal_(w2, mean=0.0, std=0.04)
    return {
        "w1": nn.Parameter(w1),
        "v1": nn.Parameter(v1),
        "w2": nn.Parameter(w2),
        "alpha_1": torch.full((1, cfg.nb_hidden), alpha_h, device=device, dtype=dtype),
        "beta_1": torch.full((1, cfg.nb_hidden), beta_h, device=device, dtype=dtype),
        "thr": torch.ones((1, cfg.nb_hidden), device=device, dtype=dtype),
        "reset": torch.zeros((1, cfg.nb_hidden), device=device, dtype=dtype),
        "rest": torch.zeros((1, cfg.nb_hidden), device=device, dtype=dtype),
        "alpha_2": torch.full((1, cfg.nb_outputs), alpha_o, device=device, dtype=dtype),
        "beta_2": torch.full((1, cfg.nb_outputs), beta_o, device=device, dtype=dtype),
    }


def state_trainable_parameters(state: Dict[str, torch.Tensor]) -> List[torch.nn.Parameter]:
    return [v for v in state.values() if isinstance(v, torch.nn.Parameter) and v.requires_grad]


def set_primary_weight_training(state: Dict[str, torch.Tensor], trainable: bool):
    for name in ("w1", "v1", "w2"):
        tensor = state.get(name)
        if isinstance(tensor, torch.nn.Parameter):
            tensor.requires_grad_(bool(trainable))


def snapshot_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items() if isinstance(tensor, torch.Tensor)}


def state_param_counts(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    trainable = 0
    frozen = 0
    for tensor in state.values():
        if not isinstance(tensor, torch.Tensor):
            continue
        count = int(tensor.numel())
        if isinstance(tensor, torch.nn.Parameter) and tensor.requires_grad:
            trainable += count
        else:
            frozen += count
    return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}


def _checkpoint_params_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "primary_state" in ckpt:
        return ckpt["primary_state"]
    if "params" in ckpt:
        return ckpt["params"]
    if "snn_params" in ckpt:
        return ckpt["snn_params"]
    return ckpt


def _param_from_aliases(params: Dict[str, torch.Tensor], aliases: List[str]) -> Optional[torch.Tensor]:
    for name in aliases:
        value = params.get(name)
        if value is not None:
            return value
    return None


def _coerce_param_shape(name: str, tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().to(device=device, dtype=dtype).clone()
    if tuple(value.shape) == tuple(target.shape):
        return value
    if value.numel() == target.numel():
        return value.reshape_as(target)
    raise ValueError(
        f"Checkpoint tensor {name} has shape {tuple(value.shape)}, expected {tuple(target.shape)} "
        f"for this release-run config."
    )


def load_base_snn_state(
    ckpt_path: Union[str, Path],
    cfg: ReleaseConfig,
    train_snn: bool,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    ckpt_path = Path(os.path.expanduser(str(ckpt_path)))
    ckpt = torch.load(ckpt_path, map_location=device)
    params = _checkpoint_params_dict(ckpt)
    if not isinstance(params, dict):
        raise ValueError(f"Checkpoint {ckpt_path} does not contain a compatible parameter dictionary.")

    state = setup_primary_state(cfg)
    aliases = {
        "w1": ["w1"],
        "v1": ["v1"],
        "w2": ["w2"],
        "alpha_1": ["alpha_1", "alpha_hetero_1", "alpha"],
        "beta_1": ["beta_1", "beta_hetero_1", "beta"],
        "thr": ["thr", "thresholds_1", "threshold"],
        "reset": ["reset", "reset_1"],
        "rest": ["rest", "rest_1"],
        "alpha_2": ["alpha_2", "alpha_hetero_2"],
        "beta_2": ["beta_2", "beta_hetero_2"],
    }
    loaded: List[str] = []
    missing: List[str] = []
    for target_name, names in aliases.items():
        tensor = _param_from_aliases(params, names)
        if tensor is None:
            missing.append(target_name)
            continue
        coerced = _coerce_param_shape(target_name, tensor, state[target_name])
        if target_name in {"w1", "v1", "w2"}:
            state[target_name] = torch.nn.Parameter(coerced, requires_grad=bool(train_snn))
        else:
            state[target_name] = coerced
        loaded.append(target_name)

    set_primary_weight_training(state, train_snn)
    metadata = {
        "path": str(ckpt_path),
        "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "metrics": ckpt.get("metrics") if isinstance(ckpt, dict) else None,
        "loaded": loaded,
        "missing": missing,
        "source_keys": sorted([str(k) for k in params.keys()]),
    }
    return state, metadata


def matrix_scale(matrix: torch.Tensor) -> torch.Tensor:
    max_abs = matrix.detach().abs().amax(dim=(1, 2), keepdim=True)
    std = matrix.detach().std(dim=(1, 2), keepdim=True, unbiased=False)
    return torch.maximum(max_abs, 3.0 * std).clamp_min(1e-6)


def field_from_releases(
    amplitudes: torch.Tensor,
    spreads: torch.Tensor,
    sites: torch.Tensor,
    n: int,
    cfg: ReleaseConfig,
) -> torch.Tensor:
    kernels = spread_kernels(n, sites.to(amplitudes.device), spreads, cfg.grid_dim, cfg.spread_mode)
    if kernels.numel() == 0:
        return amplitudes.new_zeros((amplitudes.size(0), n))
    numerator = torch.sum(amplitudes.unsqueeze(2) * kernels, dim=1)
    denom = torch.sum(kernels, dim=1).clamp_min(1e-6)
    return numerator / denom


def apply_release_weight_update(
    w1_local: torch.Tensor,
    v1_local: torch.Tensor,
    w2_local: torch.Tensor,
    release: Dict[str, torch.Tensor],
    modulator: ReleaseSiteMLP,
    cfg: ReleaseConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_field_w1 = field_from_releases(
        release["amp_w1"], release["spread_h"], modulator.hidden_sites, cfg.nb_hidden, cfg
    )
    hidden_field_v1 = field_from_releases(
        release["amp_v1"], release["spread_h"], modulator.hidden_sites, cfg.nb_hidden, cfg
    )
    output_field_w2 = field_from_releases(
        release["amp_w2"], release["spread_o"], modulator.output_sites, cfg.nb_outputs, cfg
    )

    if cfg.update_mode == "sub":
        w1_new = hidden_field_w1.unsqueeze(1) * matrix_scale(w1_local) * cfg.sub_scale
        v1_new = hidden_field_v1.unsqueeze(1) * matrix_scale(v1_local) * cfg.sub_scale
        w2_new = output_field_w2.unsqueeze(1) * matrix_scale(w2_local) * cfg.sub_scale
        return w1_new, v1_new, w2_new

    w1_new = w1_local + hidden_field_w1.unsqueeze(1) * matrix_scale(w1_local) * cfg.delta_scale
    v1_new = v1_local + hidden_field_v1.unsqueeze(1) * matrix_scale(v1_local) * cfg.delta_scale
    w2_new = w2_local + output_field_w2.unsqueeze(1) * matrix_scale(w2_local) * cfg.delta_scale
    return w1_new, v1_new, w2_new


def run_snn_weight_release_modulated(
    inputs: torch.Tensor,
    state: Dict[str, torch.Tensor],
    modulator: ReleaseSiteMLP,
    cfg: ReleaseConfig,
    modulation_events: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    batch = inputs.size(0)
    alpha_1 = state["alpha_1"].expand(batch, cfg.nb_hidden)
    beta_1 = state["beta_1"].expand(batch, cfg.nb_hidden)
    thr = state["thr"].expand(batch, cfg.nb_hidden)
    rst = state["reset"].expand(batch, cfg.nb_hidden)
    rest = state["rest"].expand(batch, cfg.nb_hidden)
    alpha_2 = state["alpha_2"].expand(batch, cfg.nb_outputs)
    beta_2 = state["beta_2"].expand(batch, cfg.nb_outputs)

    w1_local = state["w1"].unsqueeze(0).expand(batch, -1, -1).clone()
    v1_local = state["v1"].unsqueeze(0).expand(batch, -1, -1).clone()
    w2_local = state["w2"].unsqueeze(0).expand(batch, -1, -1).clone()

    syn = torch.zeros((batch, cfg.nb_hidden), device=inputs.device, dtype=inputs.dtype)
    mem = torch.zeros_like(syn)
    out_h = torch.zeros_like(syn)
    flt2 = torch.zeros((batch, cfg.nb_outputs), device=inputs.device, dtype=inputs.dtype)
    out2 = torch.zeros_like(flt2)

    spk_rec: List[torch.Tensor] = []
    mem_rec: List[torch.Tensor] = []
    out_rec: List[torch.Tensor] = []
    k = max(1, int(cfg.ann_interval))

    for t in range(cfg.nb_steps):
        h1_t = torch.einsum("bi,bih->bh", inputs[:, t, :], w1_local)
        h1_t = h1_t + torch.einsum("bh,bhr->br", out_h, v1_local)

        mthr = mem - thr
        out_h = spike_fn(mthr)
        rst_mask = (mthr > 0).float()
        syn = alpha_1 * syn + h1_t
        mem = beta_1 * (mem - rest) + rest + (1.0 - beta_1) * syn - rst_mask * (thr - rst)
        spk_rec.append(out_h)
        mem_rec.append(mem)

        h2_t = torch.einsum("bh,bho->bo", out_h, w2_local)
        flt2 = alpha_2 * flt2 + h2_t
        out2 = beta_2 * out2 + (1.0 - beta_2) * flt2
        out_rec.append(out2)

        if t % k == 0:
            start = max(0, t - k + 1)
            in_flat = inputs[:, start:t + 1, :].sum(dim=1)
            hid_flat = torch.stack(spk_rec[start:t + 1], dim=1).sum(dim=1)
            out_flat = torch.stack(out_rec[start:t + 1], dim=1).sum(dim=1)
            mlp_in = modulator.build_input(in_flat, hid_flat, out_flat, w1_local, v1_local, w2_local)
            release = modulator(mlp_in)
            w1_next, v1_next, w2_next = apply_release_weight_update(
                w1_local, v1_local, w2_local, release, modulator, cfg
            )
            if modulation_events is not None:
                modulation_events.append({
                    "step": int(t),
                    "window_start": int(start),
                    "window_size": int(t - start + 1),
                    "input_spike_sum": float(in_flat.detach().sum().item()),
                    "hidden_spike_sum": float(hid_flat.detach().sum().item()),
                    "output_mem_sum": float(out_flat.detach().sum().item()),
                    "amp_w1": tensor_summary(release["amp_w1"], include_shape=False),
                    "amp_v1": tensor_summary(release["amp_v1"], include_shape=False),
                    "amp_w2": tensor_summary(release["amp_w2"], include_shape=False),
                    "spread_h": tensor_summary(release["spread_h"], include_shape=False),
                    "spread_o": tensor_summary(release["spread_o"], include_shape=False),
                    "delta_w1": tensor_summary(w1_next - w1_local, include_shape=False),
                    "delta_v1": tensor_summary(v1_next - v1_local, include_shape=False),
                    "delta_w2": tensor_summary(w2_next - w2_local, include_shape=False),
                })
            w1_local, v1_local, w2_local = w1_next, v1_next, w2_next

    return torch.stack(out_rec, dim=1), (torch.stack(mem_rec, dim=1), torch.stack(spk_rec, dim=1))


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def spike_regularization_loss(spikes: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0.0:
        return spikes.new_zeros(())
    mean_rate = spikes.mean(dim=1)
    high_rate = torch.clamp(mean_rate - 0.01, min=0.0).pow(2).mean()
    sample_rate = spikes.mean(dim=(1, 2))
    burst = torch.clamp(sample_rate - 0.06, min=0.0).pow(2).mean()
    return float(scale) * (high_rate + burst)


@torch.no_grad()
def evaluate_release_model(
    state: Dict[str, torch.Tensor],
    modulator: ReleaseSiteMLP,
    cfg: ReleaseConfig,
    x_data,
    y_data,
    batch_size: int,
    max_time: float,
    indices: Optional[np.ndarray] = None,
    batch_limit: Optional[int] = None,
) -> Dict[str, float]:
    modulator.eval()
    loss_fn = nn.NLLLoss(reduction="mean")
    log_softmax = nn.LogSoftmax(dim=1)
    losses: List[float] = []
    accs: List[float] = []
    batches = 0
    gen = sparse_data_generator_from_hdf5_spikes(
        x_data,
        y_data,
        batch_size,
        cfg.nb_steps,
        cfg.nb_inputs,
        max_time,
        shuffle=False,
        indices=indices,
    )
    for x_local, y_local in gen:
        out, _ = run_snn_weight_release_modulated(x_local.to_dense(), state, modulator, cfg)
        m, _ = torch.max(out, dim=1)
        log_p = log_softmax(m)
        loss = loss_fn(log_p, y_local)
        pred = torch.argmax(m, dim=1)
        losses.append(float(loss.item()))
        accs.append(float((pred == y_local).float().mean().item()))
        batches += 1
        if batch_limit is not None and batches >= int(batch_limit):
            break
    return {
        "nll": float(np.mean(losses)) if losses else float("nan"),
        "acc": float(np.mean(accs)) if accs else 0.0,
        "batches": int(batches),
    }


def save_release_checkpoint(
    path: Path,
    epoch: int,
    metrics: Dict[str, float],
    history: Dict[str, List[float]],
    state: Dict[str, torch.Tensor],
    modulator: ReleaseSiteMLP,
    cfg: ReleaseConfig,
    cli_options: Dict[str, Any],
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "metrics": metrics,
            "history": history,
            "primary_state": snapshot_state(state),
            "modulator_state": modulator.state_dict(),
            "config": asdict(cfg),
            "cli_options": cli_options,
        },
        path,
    )


def train_real_run(args: argparse.Namespace, cfg: ReleaseConfig):
    set_seed(args.seed)
    run_dir = ensure_run_dir(args.save_dir)
    fold_dir = run_dir / "Fold_1"
    fold_dir.mkdir(parents=True, exist_ok=True)
    text_log_path = run_dir / "training_log.txt"
    json_log_path = Path(args.log_path).expanduser() if args.log_path else run_dir / "run_log.jsonl"

    log_file = text_log_path.open("a", buffering=1, encoding="utf-8")

    def log(message: str):
        print(message)
        print(message, file=log_file)

    def json_log(payload: Dict[str, Any]):
        if not args.no_log:
            write_run_log(str(json_log_path), payload)

    cli_options = vars(args).copy()
    prelog_messages: List[str] = []
    base_snn_ckpt = normalize_optional_path(args.base_snn_ckpt)
    base_snn_stockpile_dir = normalize_optional_path(args.base_snn_stockpile_dir)
    stockpile_split_path: Optional[str] = None
    if args.base_snn_from_stockpile:
        if base_snn_ckpt is not None:
            prelog_messages.append("[info] --base_snn_from_stockpile enabled; ignoring --base_snn_ckpt.")
        stockpile_source = base_snn_stockpile_dir or "SNN_Stockpile/Base_SNN/SNN"
        stockpile_ckpt = pick_random_stockpile_ckpt(stockpile_source, seed=args.base_snn_stockpile_seed)
        base_snn_ckpt = str(stockpile_ckpt)
        candidate_split = stockpile_ckpt.parent / "split_indices.npz"
        if candidate_split.exists():
            stockpile_split_path = str(candidate_split)
            prelog_messages.append(f"[info] Using stockpile split indices from {candidate_split}")
        else:
            prelog_messages.append(
                f"[warn] Stockpile split indices not found next to {stockpile_ckpt}; using default split settings."
            )
    fixed_split_path = stockpile_split_path or normalize_optional_path(args.fixed_split_path)
    if stockpile_split_path is not None and not args.use_validation:
        prelog_messages.append("[info] Enabling validation to reuse stockpile split indices.")
    cli_options["effective_base_snn_ckpt"] = base_snn_ckpt
    cli_options["effective_fixed_split_path"] = fixed_split_path
    cli_options["stockpile_split_path"] = stockpile_split_path

    log(f"Saving to: {run_dir}")
    log(f"Text log: {text_log_path}")
    log(f"JSONL log: {json_log_path if not args.no_log else 'disabled'}")
    log(f"device: {device}")
    for message in prelog_messages:
        log(message)
    log("---- Options ----")
    for key in sorted(cli_options):
        log(f"{key}: {cli_options[key]}")
    log("---- ReleaseConfig ----")
    for key, value in sorted(asdict(cfg).items()):
        log(f"{key}: {value}")

    x_train, y_train, x_test, y_test = load_h5(args.cache_dir, args.cache_subdir, args.train_file, args.test_file)
    x_val, y_val = load_validation_split(args.cache_dir, args.cache_subdir, args.val_file)
    external_val = x_val is not None and y_val is not None

    if external_val:
        train_idx = np.arange(len(y_train), dtype=np.int64)
        val_idx = None
        val_x, val_y = x_val, y_val
        log(f"External validation: {args.val_file}, size={len(y_val)}")
        if fixed_split_path:
            log("[info] External validation provided; ignoring fixed/stockpile train split.")
    elif fixed_split_path:
        train_idx, val_idx, split_meta = load_indices(fixed_split_path)
        val_x, val_y = (x_train, y_train) if val_idx is not None else (None, None)
        log(f"Loaded fixed split from {fixed_split_path}")
        if split_meta:
            log(f"Loaded split meta: {split_meta}")
    elif args.use_validation:
        train_idx, val_idx = stratified_split_indices(y_train, args.val_fraction, args.seed)
        val_x, val_y = x_train, y_train
        log(f"Validation split: val_fraction={args.val_fraction}")
    else:
        train_idx = np.arange(len(y_train), dtype=np.int64)
        val_idx = None
        val_x, val_y = None, None
        log("Validation disabled.")

    train_idx = maybe_limit_indices(train_idx, args.train_subset, args.seed)
    if val_idx is not None:
        val_idx = maybe_limit_indices(val_idx, args.val_subset, args.seed + 1)
    test_idx = maybe_limit_indices(np.arange(len(y_test), dtype=np.int64), args.test_subset, args.seed + 2)
    save_indices(
        fold_dir / "split_indices.npz",
        train_idx,
        val_idx,
        {
            "external_validation": bool(external_val),
            "use_validation": bool(args.use_validation),
            "val_fraction": float(args.val_fraction),
            "train_subset": args.train_subset,
            "val_subset": args.val_subset,
            "test_subset": args.test_subset,
            "fixed_split_path": fixed_split_path,
            "stockpile_split_path": stockpile_split_path,
            "base_snn_ckpt": base_snn_ckpt,
            "seed": int(args.seed),
        },
    )
    log(f"Train size: {len(train_idx)}")
    if external_val:
        log(f"Validation size: {len(y_val)}")
    elif val_idx is not None:
        log(f"Validation size: {len(val_idx)}")
    log(f"Test size: {len(test_idx)}")

    base_ckpt_metadata: Optional[Dict[str, Any]] = None
    if base_snn_ckpt is not None:
        state, base_ckpt_metadata = load_base_snn_state(base_snn_ckpt, cfg, args.train_snn)
        log("---- Base SNN Checkpoint ----")
        log(f"Path: {base_snn_ckpt}")
        if base_ckpt_metadata.get("epoch") is not None:
            log(f"Epoch: {base_ckpt_metadata['epoch']}")
        if base_ckpt_metadata.get("metrics"):
            log(f"Metrics: {base_ckpt_metadata['metrics']}")
        log(f"Loaded tensors: {', '.join(base_ckpt_metadata['loaded'])}")
        if base_ckpt_metadata.get("missing"):
            log(f"[warn] Missing tensors kept from fresh init: {', '.join(base_ckpt_metadata['missing'])}")
    else:
        state = setup_primary_state(cfg)
        set_primary_weight_training(state, args.train_snn)
        log("[info] No base SNN checkpoint provided; using random primary SNN initialization.")
    modulator = ReleaseSiteMLP(cfg).to(device)
    optimizer_params: List[torch.nn.Parameter] = list(modulator.parameters()) + state_trainable_parameters(state)
    optimizer = torch.optim.Adam(optimizer_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    loss_fn = nn.NLLLoss(reduction="mean")
    log_softmax = nn.LogSoftmax(dim=1)

    initial_payload = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "event": "run_start",
        "run_dir": str(run_dir),
        "device": str(device),
        "cli_options": cli_options,
        "config": asdict(cfg),
        "modulator": {
            "input_dim": int(modulator.input_dim),
            "output_dim": int(modulator.output_dim),
            "params": int(count_params(modulator)),
            "hidden_release_count": int(modulator.hidden_release_count),
            "output_release_count": int(modulator.output_release_count),
            "hidden_sites": [int(v) for v in modulator.hidden_sites.detach().cpu().tolist()],
            "output_sites": [int(v) for v in modulator.output_sites.detach().cpu().tolist()],
        },
        "primary_state_counts": state_param_counts(state),
        "base_snn_checkpoint": base_ckpt_metadata,
        "initial_state": {
            name: tensor_summary(tensor)
            for name, tensor in state.items()
            if isinstance(tensor, torch.Tensor)
        },
        "split_sizes": {
            "train": int(len(train_idx)),
            "validation": int(len(y_val) if external_val else (len(val_idx) if val_idx is not None else 0)),
            "test": int(len(test_idx)),
        },
    }
    json_log(initial_payload)
    log("---- Model ----")
    log(f"modulator input_dim: {modulator.input_dim}")
    log(f"modulator output_dim: {modulator.output_dim}")
    log(f"modulator params: {count_params(modulator)}")
    counts = state_param_counts(state)
    log(f"primary state params: trainable={counts['trainable']} frozen={counts['frozen']} total={counts['total']}")

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_nll": [],
        "val_acc": [],
        "test_nll": [],
        "test_acc": [],
    }
    best_metric = -float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    eval_every = max(1, int(args.eval_every))

    for epoch in range(1, int(args.nb_epochs) + 1):
        modulator.train()
        local_losses: List[float] = []
        local_accs: List[float] = []
        first_batch_events: Optional[List[Dict[str, Any]]] = None
        train_batches = 0
        gen = sparse_data_generator_from_hdf5_spikes(
            x_train,
            y_train,
            args.batch_size,
            cfg.nb_steps,
            cfg.nb_inputs,
            args.max_time,
            shuffle=True,
            indices=train_idx,
        )
        for batch_idx, (x_local, y_local) in enumerate(gen):
            event_sink = [] if (args.log_mod_events and batch_idx == 0) else None
            out, (_, spk) = run_snn_weight_release_modulated(
                x_local.to_dense(),
                state,
                modulator,
                cfg,
                event_sink,
            )
            m, _ = torch.max(out, dim=1)
            log_p = log_softmax(m)
            loss = loss_fn(log_p, y_local)
            if args.spike_reg_enable:
                loss = loss + spike_regularization_loss(spk, args.spike_reg_scale)
            pred = torch.argmax(m, dim=1)
            acc = float((pred == y_local).float().mean().item())

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(optimizer_params, float(args.grad_clip))
            optimizer.step()

            if event_sink is not None:
                first_batch_events = event_sink
            local_losses.append(float(loss.item()))
            local_accs.append(acc)
            train_batches += 1
            if args.train_batch_limit is not None and train_batches >= int(args.train_batch_limit):
                break

        train_loss = float(np.mean(local_losses)) if local_losses else float("nan")
        train_acc = float(np.mean(local_accs)) if local_accs else 0.0
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        should_eval = (epoch % eval_every == 0) or (epoch == int(args.nb_epochs))
        val_metrics = {"nll": float("nan"), "acc": 0.0, "batches": 0}
        test_metrics = {"nll": float("nan"), "acc": 0.0, "batches": 0}
        if should_eval and val_x is not None and val_y is not None:
            val_metrics = evaluate_release_model(
                state,
                modulator,
                cfg,
                val_x,
                val_y,
                args.batch_size,
                args.max_time,
                indices=None if external_val else val_idx,
                batch_limit=args.eval_batch_limit,
            )
        if should_eval:
            test_metrics = evaluate_release_model(
                state,
                modulator,
                cfg,
                x_test,
                y_test,
                args.batch_size,
                args.max_time,
                indices=test_idx,
                batch_limit=args.eval_batch_limit,
            )

        history["val_nll"].append(float(val_metrics["nll"]))
        history["val_acc"].append(float(val_metrics["acc"]))
        history["test_nll"].append(float(test_metrics["nll"]))
        history["test_acc"].append(float(test_metrics["acc"]))

        metric_for_best = float(val_metrics["acc"]) if (val_x is not None and val_y is not None) else train_acc
        metrics_payload = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_nll": float(val_metrics["nll"]),
            "val_acc": float(val_metrics["acc"]),
            "test_nll": float(test_metrics["nll"]),
            "test_acc": float(test_metrics["acc"]),
            "train_batches": int(train_batches),
            "val_batches": int(val_metrics["batches"]),
            "test_batches": int(test_metrics["batches"]),
        }
        event_payload = {
            "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "event": "epoch",
            "epoch": int(epoch),
            "metrics": metrics_payload,
            "first_batch_modulation_summary": summarize_modulation_events(first_batch_events or []),
            "first_batch_modulation_events": first_batch_events or [],
            "state": {
                name: tensor_summary(tensor)
                for name, tensor in state.items()
                if isinstance(tensor, torch.Tensor)
            },
        }
        json_log(event_payload)
        log(
            f"Epoch {epoch}: train_loss={train_loss:.5f} train_acc={train_acc:.5f} "
            f"val_nll={val_metrics['nll']:.5f} val_acc={val_metrics['acc']:.5f} "
            f"test_nll={test_metrics['nll']:.5f} test_acc={test_metrics['acc']:.5f}"
        )

        if args.save_every_epoch:
            save_release_checkpoint(
                fold_dir / f"release_epoch_{epoch}.pth",
                epoch,
                metrics_payload,
                history,
                state,
                modulator,
                cfg,
                cli_options,
            )

        if metric_for_best > best_metric + 1e-6:
            best_metric = metric_for_best
            best_epoch = epoch
            epochs_no_improve = 0
            save_release_checkpoint(
                fold_dir / "release_best.pth",
                epoch,
                metrics_payload,
                history,
                state,
                modulator,
                cfg,
                cli_options,
            )
            log(f"New best checkpoint at epoch {epoch} (metric={best_metric:.5f})")
        else:
            epochs_no_improve += 1
            if args.patience and args.patience > 0:
                log(f"No improvement ({epochs_no_improve}/{args.patience}); best epoch={best_epoch}")
                if epochs_no_improve >= int(args.patience):
                    log("Early stopping.")
                    break

    final_payload = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "event": "run_end",
        "run_dir": str(run_dir),
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "history": history,
    }
    json_log(final_payload)
    log_file.close()
    return {"run_dir": str(run_dir), "best_epoch": best_epoch, "best_metric": best_metric}


def parse_hidden_sizes(value: str) -> Tuple[int, ...]:
    text = str(value or "").strip()
    if not text:
        return tuple()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def cfg_from_args(args: argparse.Namespace) -> ReleaseConfig:
    return ReleaseConfig(
        nb_inputs=args.nb_inputs,
        nb_hidden=args.nb_hidden,
        nb_outputs=args.nb_outputs,
        nb_steps=args.nb_steps,
        time_step=args.time_step,
        tau_syn=args.tau_syn,
        tau_mem=args.tau_mem,
        ann_interval=args.ann_interval,
        release_hidden=args.release_hidden,
        release_output=args.release_output,
        grid_dim=args.grid_dim,
        spread_mode=args.spread_mode,
        summary_width=args.summary_width,
        summary_mode=args.summary_mode,
        update_mode=args.update_mode,
        delta_scale=args.delta_scale,
        sub_scale=args.sub_scale,
        hidden_sizes=parse_hidden_sizes(args.hidden_sizes),
    )


def smoke_test(
    cfg: ReleaseConfig,
    batch_size: int,
    spike_prob: float,
    seed: Optional[int],
    cli_options: Dict[str, Any],
    log_path: Optional[str],
    log_enabled: bool,
):
    if seed is not None:
        torch.manual_seed(int(seed))
    state = setup_primary_state(cfg)
    modulator = ReleaseSiteMLP(cfg).to(device)
    x = (torch.rand((batch_size, cfg.nb_steps, cfg.nb_inputs), device=device) < float(spike_prob)).to(dtype)
    modulation_events: List[Dict[str, Any]] = []
    out, (mem, spk) = run_snn_weight_release_modulated(x, state, modulator, cfg, modulation_events)
    state_stats = {
        name: tensor_summary(value)
        for name, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    result = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "event": "smoke_test",
        "device": str(device),
        "cli_options": cli_options,
        "config": asdict(cfg),
        "batch_size": int(batch_size),
        "spike_prob": float(spike_prob),
        "seed": None if seed is None else int(seed),
        "modulator": {
            "input_dim": int(modulator.input_dim),
            "output_dim": int(modulator.output_dim),
            "params": int(count_params(modulator)),
            "hidden_release_count": int(modulator.hidden_release_count),
            "output_release_count": int(modulator.output_release_count),
            "hidden_sites": [int(v) for v in modulator.hidden_sites.detach().cpu().tolist()],
            "output_sites": [int(v) for v in modulator.output_sites.detach().cpu().tolist()],
        },
        "state": state_stats,
        "input": tensor_summary(x),
        "output_membrane": tensor_summary(out),
        "hidden_membrane": tensor_summary(mem),
        "hidden_spikes": tensor_summary(spk),
        "modulation_summary": summarize_modulation_events(modulation_events),
        "modulation_events": modulation_events,
    }
    print("device:", device)
    print("config:", cfg)
    print("modulator input_dim:", modulator.input_dim)
    print("modulator output_dim:", modulator.output_dim)
    print("modulator params:", count_params(modulator))
    print("out:", tuple(out.shape), "finite:", bool(torch.isfinite(out).all()))
    print("spikes:", tuple(spk.shape), "finite:", bool(torch.isfinite(spk).all()))
    print("modulation updates:", len(modulation_events))
    if log_enabled:
        written = write_run_log(log_path, result)
        print("log:", written)


def parse_args():
    p = argparse.ArgumentParser(description="Release-site weight modulation core.")
    p.add_argument("--smoke", action="store_true", help="Debug only: run a random-input forward pass instead of training.")
    p.add_argument("--nb_inputs", type=int, default=700)
    p.add_argument("--nb_hidden", type=int, default=256)
    p.add_argument("--nb_outputs", type=int, default=20)
    p.add_argument("--nb_steps", type=int, default=100)
    p.add_argument("--time_step", type=float, default=1e-3)
    p.add_argument("--tau_syn", type=float, default=10e-3)
    p.add_argument("--tau_mem", type=float, default=20e-3)
    p.add_argument("--ann_interval", type=int, default=3)
    p.add_argument("--release_hidden", type=int, default=64)
    p.add_argument("--release_output", type=int, default=8)
    p.add_argument("--grid_dim", choices=["1d", "2d"], default="2d")
    p.add_argument("--spread_mode", choices=["uniform", "normal"], default="normal")
    p.add_argument("--summary_width", type=float, default=2.0)
    p.add_argument("--summary_mode", choices=["uniform", "normal"], default="uniform")
    p.add_argument("--update_mode", choices=["add", "sub"], default="add")
    p.add_argument("--delta_scale", type=float, default=0.05)
    p.add_argument("--sub_scale", type=float, default=5.0)
    p.add_argument("--hidden_sizes", type=str, default="[256]")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--spike_prob", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--cache_dir", type=str, default="~/data")
    p.add_argument("--cache_subdir", type=str, default="hdspikes")
    p.add_argument("--train_file", type=str, default="shd_train.h5")
    p.add_argument("--test_file", type=str, default="shd_test.h5")
    p.add_argument("--val_file", type=str, default=None)
    p.add_argument("--max_time", type=float, default=1.4)
    p.add_argument("--nb_epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--save_dir", type=str, default="Runs_WeightReleaseMod")
    p.add_argument("--use_validation", type=str2bool, nargs="?", const=True, default=True)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--fixed_split_path", type=str, default=None)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--eval_batch_limit", type=int, default=None)
    p.add_argument("--train_batch_limit", type=int, default=None)
    p.add_argument("--train_subset", type=int, default=None)
    p.add_argument("--val_subset", type=int, default=None)
    p.add_argument("--test_subset", type=int, default=None)
    p.add_argument("--train_snn", type=str2bool, nargs="?", const=True, default=True)
    p.add_argument("--save_every_epoch", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--spike_reg_enable", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--spike_reg_scale", type=float, default=1.0)
    p.add_argument("--log_mod_events", type=str2bool, nargs="?", const=True, default=True)
    p.add_argument("--base_snn_ckpt", type=str, default=None, help="Path to a base SNN checkpoint to initialize the primary SNN.")
    p.add_argument(
        "--base_snn_from_stockpile",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, sample a base SNN checkpoint from a stockpile directory and ignore --base_snn_ckpt.",
    )
    p.add_argument(
        "--base_snn_stockpile_dir",
        type=str,
        default=None,
        help="Directory containing Run_* folders with base SNN checkpoints. Default matches snn_allinone: SNN_Stockpile/Base_SNN/SNN.",
    )
    p.add_argument(
        "--base_snn_stockpile_seed",
        type=int,
        default=None,
        help="Optional seed to make stockpile checkpoint selection reproducible.",
    )
    p.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="Append a JSONL run log here. Defaults to run_dir/run_log.jsonl for real runs.",
    )
    p.add_argument("--no_log", action="store_true", help="Disable JSONL run logging.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = cfg_from_args(args)
    if args.smoke:
        smoke_test(
            cfg,
            args.batch_size,
            args.spike_prob,
            args.seed,
            vars(args),
            args.log_path,
            not args.no_log,
        )
    else:
        result = train_real_run(args, cfg)
        print("\nRelease-site weight modulation training complete.")
        print(result)
