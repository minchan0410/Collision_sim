#!/usr/bin/env python3
# run.py - batch runner for maker.py
# Usage example (mp4):
#   python run.py --in_root "C:\...\Mat" --out_root "C:\...\SBEV_OUT" --export mp4 --lane_mode accum
#
# It mirrors the directory structure of in_root into out_root.
# Skipped files are saved under: out_root/_logs/

from __future__ import annotations
import argparse
from pathlib import Path
import traceback

import Mat_Process.notusing.maker as maker  # maker.py must be in the same folder (or in PYTHONPATH)


def build_tree(paths: list[Path]) -> str:
    """
    Build a Windows-tree-like text from relative Paths.
    Example output:
      .
      ├─A
      │  └─B
      │     ├─x.mat
      │     └─y.mat
      └─C
         └─z.mat
    """
    # Build nested dict: {name: {...}} and store files with special key
    tree = {}

    for p in paths:
        parts = list(p.parts)
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node.setdefault("__files__", []).append(parts[-1])

    def _render(node: dict, prefix: str, is_last: bool, name: str) -> list[str]:
        lines = []
        if name is not None:
            connector = "└─" if is_last else "├─"
            lines.append(prefix + connector + name)
            prefix = prefix + ("   " if is_last else "│  ")

        # list dirs and files
        dirs = sorted([k for k in node.keys() if k != "__files__"])
        files = sorted(node.get("__files__", []))

        # render dirs
        for i, d in enumerate(dirs):
            last_dir = (i == len(dirs) - 1) and (len(files) == 0)
            lines.extend(_render(node[d], prefix, last_dir, d))

        # render files
        for j, f in enumerate(files):
            last_file = (j == len(files) - 1)
            connector = "└─" if last_file else "├─"
            lines.append(prefix + connector + f)

        return lines

    lines = ["."]
    # top-level
    top_dirs = sorted(tree.keys())
    for idx, d in enumerate([k for k in top_dirs if k != "__files__"]):
        lines.extend(_render(tree[d], "", idx == (len(top_dirs) - 1), d))
    # top-level files (rare)
    for f in sorted(tree.get("__files__", [])):
        lines.append("└─" + f)

    return "\n".join(lines) + "\n"


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="Input Mat root folder (the folder that contains .mat files recursively).")
    ap.add_argument("--out_root", required=True, help="Output root folder. Directory structure will be mirrored here.")

    # maker params (forwarded)
    ap.add_argument("--lane_mode", choices=["frame", "accum"], required=True)
    ap.add_argument("--lane_window", type=int, default=30, help="Only used when lane_mode=frame. Window in rendered frames.")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--res", type=float, default=0.2)
    ap.add_argument("--margin", type=float, default=15.0)

    # export mode
    ap.add_argument("--export", choices=["mp4", "png"], required=True,
                    help="mp4: creates only mp4 (no png). png: creates only pngs (no mp4).")

    # batch behavior
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    ap.add_argument("--glob", default="*.mat", help="Pattern for mat files (default: *.mat)")

    args = ap.parse_args()

    in_root = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # logs
    log_dir = out_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    skipped_tree_path = log_dir / "skipped_tree.txt"
    skipped_details_path = log_dir / "skipped_details.txt"
    summary_path = log_dir / "run_summary.txt"

    mat_files = sorted(in_root.rglob(args.glob))

    skipped: list[Path] = []
    skipped_details: list[tuple[Path, str]] = []
    ok = 0
    already = 0

    # config for maker
    cfg = maker.Cfg(
        res=args.res,
        margin=args.margin,
        stride=args.stride,
    )

    for mat_path in mat_files:
        rel = mat_path.relative_to(in_root)
        rel_dir = rel.parent
        stem = mat_path.stem

        try:
            if args.export == "mp4":
                # output mp4 goes directly under mirrored dir
                out_dir = out_root / rel_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"{stem}.mp4"
                if out_file.exists() and not args.overwrite:
                    already += 1
                    continue

                maker.generate(
                    mat_path=str(mat_path),
                    out_dir=str(out_dir),
                    cfg=cfg,
                    lane_mode=args.lane_mode,
                    lane_window=args.lane_window,
                    export="mp4",
                    video_name=f"{stem}.mp4",
                )

            else:  # png
                # png needs a per-mat folder to avoid name collisions
                out_dir = out_root / rel_dir / stem
                out_dir.mkdir(parents=True, exist_ok=True)
                # a marker file to detect existing output
                marker = out_dir / "summary_world.png"
                if marker.exists() and not args.overwrite:
                    already += 1
                    continue

                maker.generate(
                    mat_path=str(mat_path),
                    out_dir=str(out_dir),
                    cfg=cfg,
                    lane_mode=args.lane_mode,
                    lane_window=args.lane_window,
                    export="png",
                    video_name=f"{stem}.mp4",  # ignored in png mode
                )

            ok += 1

        except Exception as e:
            skipped.append(rel)
            msg = f"{type(e).__name__}: {str(e)}"
            skipped_details.append((rel, msg))

    # write logs
    skipped_tree_path.write_text(build_tree(skipped), encoding="utf-8")

    with skipped_details_path.open("w", encoding="utf-8") as f:
        f.write("relative_path\treason\n")
        for p, reason in skipped_details:
            f.write(f"{p.as_posix()}\t{reason}\n")

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"in_root: {in_root}\n")
        f.write(f"out_root: {out_root}\n")
        f.write(f"export: {args.export}\n")
        f.write(f"lane_mode: {args.lane_mode}\n")
        f.write(f"lane_window: {args.lane_window}\n")
        f.write(f"stride: {args.stride}\n")
        f.write(f"res: {args.res}\n")
        f.write(f"margin: {args.margin}\n")
        f.write("\n")
        f.write(f"total_mat_files: {len(mat_files)}\n")
        f.write(f"processed_ok: {ok}\n")
        f.write(f"skipped: {len(skipped)}\n")
        f.write(f"already_exists_skipped: {already}\n")
        f.write("\n")
        f.write(f"skipped_tree: {skipped_tree_path}\n")
        f.write(f"skipped_details: {skipped_details_path}\n")

    print("[DONE]")
    print(f"  total: {len(mat_files)}")
    print(f"  ok: {ok}")
    print(f"  skipped: {len(skipped)} (see {skipped_tree_path})")
    print(f"  already exists skipped: {already}")
    print(f"  logs: {log_dir}")


if __name__ == "__main__":
    main()