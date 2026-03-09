#!/usr/bin/env python3
# check_mp4_and_traffic_meta.py
# Accurate + fast(ish) metadata scan for MAT v5 (incl. miCOMPRESSED) and v7.3(HDF5)

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import os, re, mmap, zlib, struct, sys
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

# v7.3(HDF5) support
try:
    import h5py  # type: ignore
except Exception:
    h5py = None

# ------------------- What maker.py needs -------------------
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

TRAFFIC_TX_RE = re.compile(r"^(Traffic_T\d\d)_tx$")

# ------------------- MAT v5 constants -------------------
# Data types (MAT-file level 5)
miINT8       = 1
miUINT8      = 2
miINT32      = 5
miUINT32     = 6
miMATRIX     = 14
miCOMPRESSED = 15

# Array classes (mxCLASS)
mxSTRUCT_CLASS = 2


def is_hdf5(path: Path) -> bool:
    """Check HDF5 signature (MAT v7.3)."""
    try:
        with path.open("rb") as f:
            sig = f.read(8)
        return sig == b"\x89HDF\r\n\x1a\n"
    except Exception:
        return False


def top_folder(rel_posix: str) -> str:
    parts = rel_posix.split("/")
    return parts[0] if len(parts) > 1 else "."


def build_tree(rel_paths: List[Path]) -> str:
    tree: Dict[str, dict] = {}
    for p in rel_paths:
        parts = list(p.parts)
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node.setdefault("__files__", []).append(parts[-1])

    def _render(node: dict, prefix: str, is_last: bool, name: str) -> List[str]:
        lines = []
        connector = "└─" if is_last else "├─"
        lines.append(prefix + connector + name)
        prefix2 = prefix + ("   " if is_last else "│  ")

        dirs = sorted([k for k in node.keys() if k != "__files__"])
        files = sorted(node.get("__files__", []))

        for i, d in enumerate(dirs):
            last_dir = (i == len(dirs) - 1) and (len(files) == 0)
            lines.extend(_render(node[d], prefix2, last_dir, d))

        for j, f in enumerate(files):
            last_file = (j == len(files) - 1)
            connector2 = "└─" if last_file else "├─"
            lines.append(prefix2 + connector2 + f)
        return lines

    out = ["."]
    top_dirs = sorted([k for k in tree.keys() if k != "__files__"])
    for i, d in enumerate(top_dirs):
        out.extend(_render(tree[d], "", i == (len(top_dirs) - 1), d))
    for f in sorted(tree.get("__files__", [])):
        out.append("└─" + f)
    return "\n".join(out) + "\n"


# ------------------- v7.3(HDF5): get data fields -------------------
def v73_get_data_fields(path: Path) -> Tuple[Optional[Set[str]], str]:
    if h5py is None:
        return None, "h5py_not_installed"
    try:
        with h5py.File(str(path), "r") as f:
            if "data" not in f:
                return None, "no_/data"
            obj = f["data"]
            if hasattr(obj, "keys"):
                return set(obj.keys()), "ok_v73_group"
            if hasattr(obj, "attrs") and "MATLAB_fields" in obj.attrs:
                raw = obj.attrs["MATLAB_fields"]
                fields = set([x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in raw])
                return fields, "ok_v73_attr"
            return None, "unknown_v73_layout"
    except Exception as e:
        return None, f"v73_error:{type(e).__name__}:{e}"


def traffic_pairs_from_fields(fields: Set[str]) -> int:
    ids = set()
    for k in fields:
        m = TRAFFIC_TX_RE.match(k)
        if m:
            tid = m.group(1)
            if f"{tid}_ty" in fields:
                ids.add(tid)
    return len(ids)


# ------------------- MAT v5 minimal parser -------------------
def _pad8(n: int) -> int:
    return (8 - (n % 8)) % 8


def _read_u32(b: bytes, off: int) -> int:
    return struct.unpack_from("<I", b, off)[0]


class _BytesReader:
    """Simple buffered reader over bytes (for decompressed streams)."""
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def read(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise EOFError("read beyond buffer")
        out = self.data[self.pos:self.pos+n]
        self.pos += n
        return out

    def skip(self, n: int) -> None:
        self.pos = min(len(self.data), self.pos + n)

    def tell(self) -> int:
        return self.pos


def _read_tag(reader: _BytesReader) -> Tuple[int, int, Optional[bytes]]:
    """
    Returns (dtype, nbytes, small_data_bytes_or_None)
    Supports small data element format.
    """
    tag = reader.read(8)
    w0 = _read_u32(tag, 0)
    w1 = _read_u32(tag, 4)

    # small data element format:
    dtype = w0 & 0xFFFF
    nbytes = (w0 >> 16) & 0xFFFF
    if nbytes != 0 and nbytes <= 4 and dtype in (miINT8, miUINT8, miINT32, miUINT32):
        data = tag[4:4+nbytes]
        return dtype, nbytes, data

    # normal format: w0=dtype, w1=nbytes
    return w0, w1, None


def _read_data(reader: _BytesReader, dtype: int, nbytes: int, small: Optional[bytes]) -> bytes:
    if small is not None:
        return small
    data = reader.read(nbytes)
    reader.skip(_pad8(nbytes))
    return data


def _parse_miMATRIX_for_struct_fields(reader: _BytesReader) -> Tuple[Optional[str], Optional[Set[str]]]:
    """
    Parse miMATRIX element body and return:
      (array_name, fieldnames_set) if this is struct array,
      otherwise (array_name, None).
    """
    # Array Flags
    dt, nb, small = _read_tag(reader)
    flags_raw = _read_data(reader, dt, nb, small)
    if len(flags_raw) < 8:
        return None, None
    flags0 = _read_u32(flags_raw, 0)
    mxclass = flags0 & 0xFF

    # Dimensions Array
    dt, nb, small = _read_tag(reader)
    _ = _read_data(reader, dt, nb, small)  # ignore

    # Array Name
    dt, nb, small = _read_tag(reader)
    name_raw = _read_data(reader, dt, nb, small)
    arr_name = name_raw.decode("utf-8", errors="ignore")

    if mxclass != mxSTRUCT_CLASS:
        return arr_name, None

    # Field name length (int32)
    dt, nb, small = _read_tag(reader)
    fnlen_raw = _read_data(reader, dt, nb, small)
    if len(fnlen_raw) < 4:
        return arr_name, None
    field_name_len = struct.unpack_from("<i", fnlen_raw, 0)[0]
    if field_name_len <= 0:
        return arr_name, None

    # Field names (int8)
    dt, nb, small = _read_tag(reader)
    fns_raw = _read_data(reader, dt, nb, small)
    nfields = nb // field_name_len if field_name_len > 0 else 0
    fset = set()
    for i in range(nfields):
        chunk = fns_raw[i*field_name_len:(i+1)*field_name_len]
        chunk = chunk.split(b"\x00", 1)[0]
        s = chunk.decode("utf-8", errors="ignore")
        if s:
            fset.add(s)

    return arr_name, fset


def v5_extract_data_fields(path: Path, target_var: str = "data") -> Tuple[Optional[Set[str]], str]:
    """
    Extract fieldnames of struct variable named `target_var` from MAT v5 file.
    Handles miCOMPRESSED by zlib-decompressing the miMATRIX.
    Stops scanning as soon as it finds the target struct fieldnames.
    """
    try:
        with path.open("rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                pos = 128  # skip header
                n = len(mm)

                while pos + 8 <= n:
                    dt = struct.unpack_from("<I", mm, pos)[0]
                    nb = struct.unpack_from("<I", mm, pos + 4)[0]
                    pos += 8

                    if dt == miCOMPRESSED:
                        comp = mm[pos:pos+nb]
                        pos += nb
                        pos += _pad8(nb)

                        try:
                            decomp = zlib.decompress(comp)
                        except Exception:
                            continue

                        r = _BytesReader(decomp)
                        try:
                            dt2, nb2, small2 = _read_tag(r)
                        except Exception:
                            continue
                        if dt2 != miMATRIX:
                            continue

                        body = r.read(nb2)
                        rr = _BytesReader(body)
                        arr_name, fset = _parse_miMATRIX_for_struct_fields(rr)
                        if arr_name == target_var and fset is not None:
                            return fset, "ok_v5_compressed"

                    elif dt == miMATRIX:
                        body = mm[pos:pos+nb]
                        pos += nb
                        pos += _pad8(nb)

                        rr = _BytesReader(body)
                        arr_name, fset = _parse_miMATRIX_for_struct_fields(rr)
                        if arr_name == target_var and fset is not None:
                            return fset, "ok_v5_uncompressed"

                    else:
                        pos += nb
                        pos += _pad8(nb)

                return None, "no_data_struct_found"

            finally:
                mm.close()

    except Exception as e:
        return None, f"v5_error:{type(e).__name__}:{e}"


# ------------------- One-file scan -------------------
def scan_one(abs_path: str, rel_path: str, target_var: str) -> Tuple[str, str, str]:
    p = Path(abs_path)

    # v7.3
    if is_hdf5(p):
        fields, status = v73_get_data_fields(p)
        if fields is None:
            return rel_path, "SKIP_UNREADABLE", status

        missing = [k for k in REQUIRED_FIELDS if k not in fields]
        if missing:
            return rel_path, "SKIP_MISSING_REQUIRED", "missing_required:" + ",".join(missing) + f" ({status})"

        tr_cnt = traffic_pairs_from_fields(fields)
        if tr_cnt < 1:
            return rel_path, "SKIP_NO_TRAFFIC", f"no_traffic_pairs={tr_cnt} ({status})"

        return rel_path, "PASS", f"ok traffic_pairs={tr_cnt} ({status})"

    # v5
    fields, status = v5_extract_data_fields(p, target_var=target_var)
    if fields is None:
        return rel_path, "SKIP_MISSING_REQUIRED", status

    missing = [k for k in REQUIRED_FIELDS if k not in fields]
    if missing:
        return rel_path, "SKIP_MISSING_REQUIRED", "missing_required:" + ",".join(missing) + f" ({status})"

    tr_cnt = traffic_pairs_from_fields(fields)
    if tr_cnt < 1:
        return rel_path, "SKIP_NO_TRAFFIC", f"no_traffic_pairs={tr_cnt} ({status})"

    return rel_path, "PASS", f"ok traffic_pairs={tr_cnt} ({status})"


# ------------------- Progress printing -------------------
def progress_line(done: int, total: int, folder: str, fdone: int, ftotal: int, pass_cnt: int) -> str:
    op = 100.0 * done / max(total, 1)
    fp = 100.0 * fdone / max(ftotal, 1)
    return f"[{done}/{total}] {op:6.2f}% | PASS={pass_cnt} | folder={folder} [{fdone}/{ftotal}] {fp:6.2f}%"


# ------------------- Main -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--glob", default="*.mat")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--progress_every", type=int, default=200)
    ap.add_argument("--target_var", default="data", help="Struct variable name to inspect (default: data)")
    args = ap.parse_args()

    in_root = Path(args.in_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mats = sorted(in_root.rglob(args.glob))
    total = len(mats)

    folder_total: Dict[str, int] = {}
    for p in mats:
        r = p.relative_to(in_root).as_posix()
        f = top_folder(r)
        folder_total[f] = folder_total.get(f, 0) + 1
    folder_done: Dict[str, int] = {k: 0 for k in folder_total.keys()}

    passed: List[str] = []
    miss_req: List[Path] = []
    no_tr: List[Path] = []
    unread: List[Path] = []
    details: List[Tuple[str, str, str]] = []

    done = 0
    last_print = ""

    active_tasks = {}
    MAX_QUEUE = args.workers * 2
    tasks_iter = iter(mats)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        while active_tasks or tasks_iter is not None:
            # 1. 큐의 빈자리만큼만 새로운 작업을 제출 (tasks_iter가 None이 아닐 때만)
            while tasks_iter is not None and len(active_tasks) < MAX_QUEUE:
                try:
                    p = next(tasks_iter)
                    r = p.relative_to(in_root).as_posix()
                    fut = ex.submit(scan_one, str(p), r, args.target_var)
                    active_tasks[fut] = r
                except StopIteration:
                    tasks_iter = None
                    break
            
            if not active_tasks:
                break

            # 2. 작업 완료 대기 (타임아웃 감시)
            done_futs, _ = wait(active_tasks.keys(), timeout=30.0, return_when=FIRST_COMPLETED)

            if not done_futs:
                print("\n[CRITICAL WARNING] 파서가 무한 대기 상태에 빠졌습니다!")
                print("현재 처리 중이던 의심 파일 목록:")
                for f, rel_path in active_tasks.items():
                    print(f" -> {rel_path}")
                print("\n위 파일들 중 하나가 손상되었거나 구조적 오류를 일으켰습니다. 스크립트를 강제 종료합니다.")
                ex.shutdown(wait=False, cancel_futures=True)
                sys.exit(1)

            # 3. 완료된 작업 처리
            for fut in done_futs:
                rel = active_tasks.pop(fut)
                try:
                    rel_ret, verdict, reason = fut.result()
                except Exception as e:
                    rel_ret, verdict, reason = rel, "SKIP_UNREADABLE", f"worker_crash:{type(e).__name__}"
                
                details.append((rel_ret, verdict, reason))
                done += 1
                f = top_folder(rel_ret)
                folder_done[f] = folder_done.get(f, 0) + 1

                if verdict == "PASS":
                    passed.append(rel_ret)
                elif verdict == "SKIP_NO_TRAFFIC":
                    no_tr.append(Path(rel_ret))
                elif verdict == "SKIP_UNREADABLE":
                    unread.append(Path(rel_ret))
                else:
                    miss_req.append(Path(rel_ret))

                if done % args.progress_every == 0 or done == total:
                    pass_cnt = len(passed)
                    line = progress_line(done, total, f, folder_done[f], folder_total[f], pass_cnt)
                    if line != last_print:
                        print(line, flush=True)
                        last_print = line

    # Write outputs
    (out_dir / "pass_list.txt").write_text("\n".join(sorted(passed)) + "\n", encoding="utf-8")
    (out_dir / "missing_required_tree.txt").write_text(build_tree(sorted(miss_req)), encoding="utf-8")
    (out_dir / "no_traffic_tree.txt").write_text(build_tree(sorted(no_tr)), encoding="utf-8")
    (out_dir / "unreadable_tree.txt").write_text(build_tree(sorted(unread)), encoding="utf-8")

    with (out_dir / "skipped_details.tsv").open("w", encoding="utf-8") as file:
        file.write("relative_path\tverdict\treason\n")
        for rel, verdict, reason in sorted(details):
            if verdict != "PASS":
                file.write(f"{rel}\t{verdict}\t{reason}\n")

    with (out_dir / "summary.txt").open("w", encoding="utf-8") as file:
        file.write(f"in_root: {in_root}\n")
        file.write(f"glob: {args.glob}\n")
        file.write(f"target_var: {args.target_var}\n")
        file.write(f"total_mat_files: {len(mats)}\n")
        file.write(f"PASS (required + traffic>=1): {len(passed)}\n")
        file.write(f"SKIP_MISSING_REQUIRED: {len(miss_req)}\n")
        file.write(f"SKIP_NO_TRAFFIC: {len(no_tr)}\n")
        file.write(f"SKIP_UNREADABLE: {len(unread)}\n")
        file.write(f"workers: {args.workers}\n")
        file.write(f"progress_every: {args.progress_every}\n")

    print("[DONE]")
    print(f"  total: {len(mats)}")
    print(f"  PASS: {len(passed)}")
    print(f"  missing_required: {len(miss_req)}")
    print(f"  no_traffic: {len(no_tr)}")
    print(f"  unreadable: {len(unread)}")
    print(f"  logs: {out_dir}")


if __name__ == "__main__":
    main()