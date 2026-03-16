#!/usr/bin/env python3
"""
Convert MAT driving scenarios into MID raw txt trajectories.

Output format per line:
  frame_id<TAB>track_id<TAB>pos_x<TAB>pos_y<TAB>yaw_rad

Design choices for MID compatibility:
- Downsample each MAT sequence to one point every target-dt seconds.
- Write raw frame ids as 0, 10, 20, ... by default for compatibility.
- Only files listed in pass_list are processed.
- All output txt files are written into a single flat folder.

Agents exported:
- Ego vehicle from Car_Con_tx / Car_Con_ty
- Traffic agents from Traffic_Txx_tx / Traffic_Txx_ty

Because MID's process_data.py assumes one track_id is temporally contiguous,
traffic tracks are automatically split into contiguous segments when:
- the slot becomes invalid,
- sampled frames are missing,
- or a large position jump is detected.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import re
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

try:
    import scipy.io as sio  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scipy is required to read MAT v5 files") from exc

try:
    import h5py  # type: ignore
except Exception:
    h5py = None


REQUIRED_FIELDS = [
    "Time",
    "Car_Con_tx",
    "Car_Con_ty",
    "Car_Yaw",
    "Sensor_Road_Road_Path_tx",
    "Sensor_Road_Road_Path_ty",
    "Sensor_Road_Road_Path_DevAng",
    "Sensor_Road_Road_Path_DevDist",
    "Vhcl_tRoad",
]

TRAFFIC_TX_RE = re.compile(r"^Traffic_T(\d+)_tx$")


# ---------------------------------------------------------------------------
# MAT access helpers
# ---------------------------------------------------------------------------

def is_hdf5(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(8) == b"\x89HDF\r\n\x1a\n"
    except Exception:
        return False


class MatAccessor:
    def keys(self) -> List[str]:
        raise NotImplementedError

    def has(self, key: str) -> bool:
        return key in self.keys()

    def get(self, key: str) -> np.ndarray:
        raise NotImplementedError


class V5Accessor(MatAccessor):
    def __init__(self, data_obj):
        self.data_obj = data_obj
        self._keys = list(getattr(data_obj, "_fieldnames", []))

    def keys(self) -> List[str]:
        return self._keys

    def get(self, key: str) -> np.ndarray:
        if key not in self._keys:
            raise KeyError(key)
        obj = getattr(self.data_obj, key)
        if hasattr(obj, "data"):
            arr = np.asarray(obj.data)
        else:
            arr = np.asarray(obj)
        return np.squeeze(arr)


class V73Accessor(MatAccessor):
    def __init__(self, path: Path):
        if h5py is None:
            raise RuntimeError("h5py is required to read MAT v7.3 files")
        self._file = h5py.File(str(path), "r")
        if "data" not in self._file:
            self._file.close()
            raise KeyError("/data")
        self._data = self._file["data"]
        if not hasattr(self._data, "keys"):
            self._file.close()
            raise TypeError("MAT v7.3 /data is not an HDF5 group")
        self._keys = list(self._data.keys())

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass

    def keys(self) -> List[str]:
        return self._keys

    def _read_h5_node(self, node) -> np.ndarray:
        if isinstance(node, h5py.Group):
            if "data" in node:
                return self._read_h5_node(node["data"])
            if len(node.keys()) == 1:
                only_key = next(iter(node.keys()))
                return self._read_h5_node(node[only_key])
            raise KeyError(f"Cannot decode HDF5 group with keys {list(node.keys())}")

        if isinstance(node, h5py.Dataset):
            value = node[()]
            if getattr(value, "dtype", None) is not None and value.dtype.kind == "O":
                refs = np.asarray(value).reshape(-1)
                decoded: List[np.ndarray] = []
                for ref in refs:
                    decoded.append(self._read_h5_node(self._file[ref]))
                if len(decoded) == 1:
                    return np.squeeze(decoded[0])
                return np.squeeze(np.array(decoded, dtype=object))
            return np.squeeze(np.asarray(value))

        return np.squeeze(np.asarray(node))

    def get(self, key: str) -> np.ndarray:
        if key not in self._keys:
            raise KeyError(key)
        return self._read_h5_node(self._data[key])


def load_mat_accessor(path: Path) -> MatAccessor:
    if is_hdf5(path):
        return V73Accessor(path)

    mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if "data" not in mat:
        raise KeyError("MAT file does not contain variable 'data'")
    return V5Accessor(mat["data"])


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def as_numeric_1d(arr: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(arr).squeeze()
    if out.ndim == 0:
        out = out.reshape(1)
    if out.ndim != 1:
        raise ValueError(f"Signal {name} is not 1D after squeeze: shape={out.shape}")
    if out.dtype.kind in {"U", "S", "O"}:
        try:
            out = out.astype(float)
        except Exception as exc:
            raise ValueError(f"Signal {name} is non-numeric") from exc
    return out


@dataclass
class TrackSegment:
    track_id: int
    frames: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    source: str

    @property
    def num_points(self) -> int:
        return int(self.frames.size)


@dataclass
class ConversionResult:
    source_rel: str
    output_name: str
    num_rows: int
    num_tracks: int
    num_ego_tracks: int
    num_traffic_tracks: int
    sampled_steps: int
    original_steps: int
    original_dt: float
    status: str
    message: str


def build_sample_indices(time_array: np.ndarray, target_dt: float) -> Tuple[np.ndarray, float]:
    time = as_numeric_1d(time_array, name="Time")
    if time.size < 2:
        return np.arange(time.size, dtype=int), float("nan")

    diffs = np.diff(time)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return np.arange(time.size, dtype=int), float("nan")

    original_dt = float(np.median(diffs))

    sample_times = np.arange(float(time[0]), float(time[-1]) + 1e-9, target_dt)
    if sample_times.size == 0:
        return np.array([0], dtype=int), original_dt

    right = np.searchsorted(time, sample_times, side="left")
    right = np.clip(right, 0, len(time) - 1)
    left = np.clip(right - 1, 0, len(time) - 1)

    use_left = np.abs(time[left] - sample_times) <= np.abs(time[right] - sample_times)
    indices = np.where(use_left, left, right)
    indices = np.unique(indices.astype(int))
    return indices, original_dt


def safe_output_name(source_rel: str) -> str:
    no_suffix = str(Path(source_rel).with_suffix(""))
    return no_suffix.replace("/", "__").replace("\\", "__") + ".txt"


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def build_yaw_series(
    x: np.ndarray,
    y: np.ndarray,
    valid_mask: np.ndarray,
    *,
    raw_yaw: Optional[np.ndarray] = None,
    speed_eps: float = 1.0e-3,
) -> np.ndarray:
    """
    Build a finite yaw sequence in radians.
    Priority:
    1) use provided raw yaw where finite,
    2) otherwise use motion direction from consecutive (x, y),
    3) otherwise forward/backward fill.
    """
    n = int(x.size)
    if n == 0:
        return np.zeros((0,), dtype=float)

    yaw = np.full((n,), np.nan, dtype=float)

    if raw_yaw is not None:
        raw = as_numeric_1d(raw_yaw, name="raw_yaw")
        if raw.size != n:
            raise ValueError(f"raw_yaw length mismatch: expected {n}, got {raw.size}")
        raw = raw.astype(float)
        finite_raw = np.isfinite(raw)
        yaw[finite_raw] = wrap_to_pi(raw[finite_raw])

    motion_yaw = np.full((n,), np.nan, dtype=float)
    if n >= 2:
        dx = np.diff(x)
        dy = np.diff(y)
        speed = np.hypot(dx, dy)
        valid_pair = (
            valid_mask[:-1]
            & valid_mask[1:]
            & np.isfinite(dx)
            & np.isfinite(dy)
            & (speed > speed_eps)
        )
        motion_yaw[:-1][valid_pair] = np.arctan2(dy[valid_pair], dx[valid_pair])
        motion_yaw[-1] = motion_yaw[-2]

    missing = ~np.isfinite(yaw)
    yaw[missing] = motion_yaw[missing]

    finite = np.isfinite(yaw)
    if not np.any(finite):
        yaw.fill(0.0)
        return yaw

    first_valid = int(np.flatnonzero(finite)[0])
    yaw[:first_valid] = yaw[first_valid]
    prev = float(yaw[first_valid])
    for i in range(first_valid + 1, n):
        if np.isfinite(yaw[i]):
            prev = float(yaw[i])
        else:
            yaw[i] = prev

    return wrap_to_pi(yaw)


def split_contiguous_segments(
    frames: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    yaw: np.ndarray,
    valid_mask: np.ndarray,
    *,
    raw_frame_step: int,
    jump_threshold_m: float,
    min_points: int,
    track_id_start: int,
    source_label: str,
) -> Tuple[List[TrackSegment], int]:
    assert frames.shape == x.shape == y.shape == yaw.shape == valid_mask.shape

    segments: List[TrackSegment] = []
    n = len(frames)
    next_track_id = track_id_start
    i = 0

    while i < n:
        if not valid_mask[i]:
            i += 1
            continue

        start = i
        end = i + 1
        while end < n and valid_mask[end]:
            frame_gap = int(frames[end] - frames[end - 1])
            if frame_gap != raw_frame_step:
                break
            jump = math.hypot(float(x[end] - x[end - 1]), float(y[end] - y[end - 1]))
            if jump > jump_threshold_m:
                break
            end += 1

        if (end - start) >= min_points:
            segments.append(
                TrackSegment(
                    track_id=next_track_id,
                    frames=frames[start:end].copy(),
                    x=x[start:end].copy(),
                    y=y[start:end].copy(),
                    yaw=yaw[start:end].copy(),
                    source=source_label,
                )
            )
            next_track_id += 1

        i = end

    return segments, next_track_id


def convert_one_file(
    path: Path,
    source_rel: str,
    *,
    out_dir: Path,
    target_dt: float,
    raw_frame_step: int,
    jump_threshold_m: float,
    min_points_per_track: int,
) -> ConversionResult:
    accessor: Optional[MatAccessor] = None
    try:
        accessor = load_mat_accessor(path)
        missing = [field for field in REQUIRED_FIELDS if not accessor.has(field)]
        if missing:
            return ConversionResult(
                source_rel=source_rel,
                output_name=safe_output_name(source_rel),
                num_rows=0, num_tracks=0, num_ego_tracks=0, num_traffic_tracks=0,
                sampled_steps=0, original_steps=0, original_dt=float("nan"),
                status="SKIP_MISSING_REQUIRED",
                message="missing fields: " + ", ".join(missing),
            )

        time = as_numeric_1d(accessor.get("Time"), name="Time")
        sample_idx, original_dt = build_sample_indices(time, target_dt=target_dt)
        if sample_idx.size < 2:
            return ConversionResult(
                source_rel=source_rel,
                output_name=safe_output_name(source_rel),
                num_rows=0, num_tracks=0, num_ego_tracks=0, num_traffic_tracks=0,
                sampled_steps=int(sample_idx.size), original_steps=int(time.size), original_dt=original_dt,
                status="SKIP_TOO_SHORT",
                message="sequence shorter than two sampled steps",
            )

        raw_frames = np.arange(sample_idx.size, dtype=int) * int(raw_frame_step)

        rows: List[Tuple[int, int, float, float, float]] = []
        next_track_id = 1
        num_ego_tracks = 0
        num_traffic_tracks = 0

        # Ego vehicle
        ego_x = as_numeric_1d(accessor.get("Car_Con_tx"), name="Car_Con_tx")[sample_idx]
        ego_y = as_numeric_1d(accessor.get("Car_Con_ty"), name="Car_Con_ty")[sample_idx]
        ego_valid = np.isfinite(ego_x) & np.isfinite(ego_y)
        ego_raw_yaw = as_numeric_1d(accessor.get("Car_Yaw"), name="Car_Yaw")[sample_idx]
        ego_yaw = build_yaw_series(ego_x, ego_y, ego_valid, raw_yaw=ego_raw_yaw)
        ego_segments, next_track_id = split_contiguous_segments(
            frames=raw_frames, x=ego_x, y=ego_y, yaw=ego_yaw, valid_mask=ego_valid,
            raw_frame_step=raw_frame_step, jump_threshold_m=jump_threshold_m,
            min_points=min_points_per_track, track_id_start=next_track_id, source_label="ego",
        )
        num_ego_tracks = len(ego_segments)
        for seg in ego_segments:
            for fr, x_val, y_val, yaw_val in zip(seg.frames, seg.x, seg.y, seg.yaw):
                rows.append((int(fr), int(seg.track_id), float(x_val), float(y_val), float(yaw_val)))

        # Traffic agents
        traffic_slots: List[Tuple[int, str, str, str]] = []
        for key in accessor.keys():
            match = TRAFFIC_TX_RE.match(key)
            if not match:
                continue
            slot_token = match.group(1)
            slot_idx = int(slot_token)
            ty_key = f"Traffic_T{slot_token}_ty"
            if accessor.has(ty_key):
                traffic_slots.append((slot_idx, slot_token, key, ty_key))
        traffic_slots.sort(key=lambda item: item[0])

        n_objs = None
        if accessor.has("Traffic_nObjs"):
            try:
                n_objs = as_numeric_1d(accessor.get("Traffic_nObjs"), name="Traffic_nObjs")[sample_idx]
            except Exception:
                n_objs = None

        for slot_idx, slot_token, tx_key, ty_key in traffic_slots:
            tx = as_numeric_1d(accessor.get(tx_key), name=tx_key)[sample_idx]
            ty = as_numeric_1d(accessor.get(ty_key), name=ty_key)[sample_idx]
            valid = np.isfinite(tx) & np.isfinite(ty)

            state_key = tx_key.replace("_tx", "_State")
            if accessor.has(state_key):
                try:
                    state = as_numeric_1d(accessor.get(state_key), name=state_key)[sample_idx]
                    valid &= state > 0
                except Exception:
                    pass

            if n_objs is not None:
                valid &= n_objs > slot_idx

            traffic_raw_yaw = None
            for yaw_key in (
                f"Traffic_T{slot_token}_rz",
                f"Traffic_T{slot_token}_Yaw",
                f"Traffic_T{slot_token}_yaw",
            ):
                if accessor.has(yaw_key):
                    try:
                        traffic_raw_yaw = as_numeric_1d(accessor.get(yaw_key), name=yaw_key)[sample_idx]
                        break
                    except Exception:
                        traffic_raw_yaw = None

            traffic_yaw = build_yaw_series(tx, ty, valid, raw_yaw=traffic_raw_yaw)

            segments, next_track_id = split_contiguous_segments(
                frames=raw_frames, x=tx, y=ty, yaw=traffic_yaw, valid_mask=valid,
                raw_frame_step=raw_frame_step, jump_threshold_m=jump_threshold_m,
                min_points=min_points_per_track, track_id_start=next_track_id,
                source_label=f"traffic_slot_{slot_idx:02d}",
            )

            num_traffic_tracks += len(segments)
            for seg in segments:
                for fr, x_val, y_val, yaw_val in zip(seg.frames, seg.x, seg.y, seg.yaw):
                    rows.append((int(fr), int(seg.track_id), float(x_val), float(y_val), float(yaw_val)))

        rows.sort(key=lambda item: (item[0], item[1]))

        out_name = safe_output_name(source_rel)
        out_path = out_dir / out_name
        with out_path.open("w", encoding="utf-8", newline="") as f:
            for frame_id, track_id, pos_x, pos_y, yaw in rows:
                f.write(f"{frame_id}\t{track_id}\t{pos_x:.6f}\t{pos_y:.6f}\t{yaw:.6f}\n")

        return ConversionResult(
            source_rel=source_rel,
            output_name=out_name,
            num_rows=len(rows),
            num_tracks=num_ego_tracks + num_traffic_tracks,
            num_ego_tracks=num_ego_tracks,
            num_traffic_tracks=num_traffic_tracks,
            sampled_steps=int(sample_idx.size),
            original_steps=int(time.size),
            original_dt=original_dt,
            status="OK",
            message="",
        )

    except Exception as exc:
        return ConversionResult(
            source_rel=source_rel,
            output_name=safe_output_name(source_rel),
            num_rows=0, num_tracks=0, num_ego_tracks=0, num_traffic_tracks=0,
            sampled_steps=0, original_steps=0, original_dt=float("nan"),
            status="ERROR",
            message=f"{type(exc).__name__}: {exc}",
        )

    finally:
        if isinstance(accessor, V73Accessor):
            accessor.close()


# ---------------------------------------------------------------------------
# Path / pass-list helpers
# ---------------------------------------------------------------------------

def read_pass_list(path: Path) -> List[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line and not line.startswith("#")]

def build_basename_index(root: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for mat_path in root.rglob("*.mat"):
        index.setdefault(mat_path.name, []).append(mat_path)
    return index

def resolve_pass_entry(root: Path, entry: str, basename_index: Dict[str, List[Path]]) -> Tuple[Optional[Path], str]:
    direct = root / entry
    if direct.exists():
        return direct, entry

    fallback = root / Path(entry).name
    if fallback.exists():
        return fallback, entry

    matches = basename_index.get(Path(entry).name, [])
    if len(matches) == 1:
        return matches[0], entry

    return None, entry


# ---------------------------------------------------------------------------
# Multiprocessing Wrapper
# ---------------------------------------------------------------------------
def _worker_wrapper(kwargs: dict, q: multiprocessing.Queue) -> None:
    try:
        res = convert_one_file(**kwargs)
        q.put(res)
    except Exception as e:
        q.put(e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert pass_list MAT files into MID raw txt format.")
    parser.add_argument("--in-root", required=True, help="Root folder that contains the MAT files.")
    parser.add_argument("--pass-list", required=True, help="Path to pass_list.txt.")
    parser.add_argument("--out-dir", required=True, help="Single flat output folder for txt files.")
    parser.add_argument("--raw-frame-step", type=int, default=10, help="Raw frame increment written to txt. Default: 10")
    parser.add_argument("--jump-threshold-m", type=float, default=20.0, help="Split track if sampled displacement exceeds this value. Default: 20.0")
    parser.add_argument("--min-points-per-track", type=int, default=2, help="Minimum sampled points per exported track. Default: 2")
    parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N files. Default: 50")
    return parser.parse_args()


def _load_data_dt_from_mat_yaml() -> float:
    config_path = Path("configs") / "mat.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Required config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML content in {config_path}")
    if "data_dt" not in cfg:
        raise KeyError(f"'data_dt' is missing in {config_path}")

    dt = float(cfg["data_dt"])
    if dt <= 0:
        raise ValueError(f"'data_dt' must be positive in {config_path}, got {dt}")
    return dt


def main() -> None:
    args = parse_args()
    in_root = Path(args.in_root).resolve()
    pass_list_path = Path(args.pass_list).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target_dt = _load_data_dt_from_mat_yaml()
    if target_dt <= 0:
        raise ValueError(f"target_dt must be positive, got {target_dt}")

    pass_entries = read_pass_list(pass_list_path)
    basename_index = build_basename_index(in_root)

    manifest_path = out_dir / "manifest.csv"

    resolved: List[Tuple[Path, str]] = []
    unresolved: List[str] = []
    for entry in pass_entries:
        path, source_rel = resolve_pass_entry(in_root, entry, basename_index)
        if path is None:
            unresolved.append(entry)
        else:
            resolved.append((path, source_rel))

    ok_count = 0
    error_count = 0
    skip_count = 0
    total_rows_written = 0
    total_tracks_written = 0

    # 프로세스 생성 방식을 'spawn'으로 강제하여 메모리 꼬임 및 C 모듈 충돌 방지
    ctx = multiprocessing.get_context('spawn')

    with manifest_path.open("w", encoding="utf-8", newline="") as f_manifest:
        writer = csv.writer(f_manifest)
        writer.writerow([
            "source_rel", "output_name", "status", "message",
            "num_rows", "num_tracks", "num_ego_tracks", "num_traffic_tracks",
            "sampled_steps", "original_steps", "original_dt",
        ])

        for idx, (mat_path, source_rel) in enumerate(resolved, start=1):
            kwargs = {
                "path": mat_path,
                "source_rel": source_rel,
                "out_dir": out_dir,
                "target_dt": float(target_dt),
                "raw_frame_step": int(args.raw_frame_step),
                "jump_threshold_m": float(args.jump_threshold_m),
                "min_points_per_track": int(args.min_points_per_track),
            }

            q = ctx.Queue()
            p = ctx.Process(target=_worker_wrapper, args=(kwargs, q))
            p.start()
            p.join()  # 자식 프로세스가 종료될 때까지 대기

            if p.exitcode != 0:
                # 자식 프로세스가 Segfault 등으로 비정상 종료된 경우
                result = ConversionResult(
                    source_rel=source_rel,
                    output_name=safe_output_name(source_rel),
                    num_rows=0, num_tracks=0, num_ego_tracks=0, num_traffic_tracks=0,
                    sampled_steps=0, original_steps=0, original_dt=float("nan"),
                    status="ERROR_CRASH",
                    message=f"Process crashed with exit code {p.exitcode} (Corrupted file or C-level Segfault)"
                )
            else:
                # 정상적으로 완료되었으나 파이썬 레벨의 Exception이 있을 수 있음
                if not q.empty():
                    out = q.get()
                    if isinstance(out, Exception):
                        result = ConversionResult(
                            source_rel=source_rel,
                            output_name=safe_output_name(source_rel),
                            num_rows=0, num_tracks=0, num_ego_tracks=0, num_traffic_tracks=0,
                            sampled_steps=0, original_steps=0, original_dt=float("nan"),
                            status="ERROR",
                            message=f"{type(out).__name__}: {out}",
                        )
                    else:
                        result = out
                else:
                    result = ConversionResult(
                        source_rel=source_rel,
                        output_name=safe_output_name(source_rel),
                        num_rows=0, num_tracks=0, num_ego_tracks=0, num_traffic_tracks=0,
                        sampled_steps=0, original_steps=0, original_dt=float("nan"),
                        status="ERROR_UNKNOWN",
                        message="Process exited normally but returned no data.",
                    )

            writer.writerow([
                result.source_rel, result.output_name, result.status, result.message,
                result.num_rows, result.num_tracks, result.num_ego_tracks,
                result.num_traffic_tracks, result.sampled_steps,
                result.original_steps, result.original_dt,
            ])
            f_manifest.flush()

            if result.status == "OK":
                ok_count += 1
                total_rows_written += result.num_rows
                total_tracks_written += result.num_tracks
            elif result.status.startswith("ERROR"):
                error_count += 1
                # 손상된 파일이 감지되면 화면에 즉시 출력하여 범인을 색출
                if "CRASH" in result.status:
                    print(f"\n[WARNING] 심각하게 손상된 파일 감지 후 건너뜀 (Segfault 우회됨): {source_rel}")
            elif result.status.startswith("SKIP"):
                skip_count += 1

            if idx % int(args.progress_every) == 0 or idx == len(resolved):
                print(f"[{idx}/{len(resolved)}] converted={ok_count} errors={error_count} skipped={skip_count}")

            del result
            gc.collect()

    print("[DONE]")
    print(f"  resolved: {len(resolved)} / {len(pass_entries)}")
    print(f"  converted: {ok_count}")
    print(f"  skipped: {skip_count}")
    print(f"  errors: {error_count}")
    print(f"  target_dt(sec): {target_dt}")
    print(f"  out_dir: {out_dir}")

if __name__ == "__main__":
    main()
