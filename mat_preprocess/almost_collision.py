#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


REQUIRED_COLUMNS = {
    "output_name",
    "status",
    "num_tracks",
    "sampled_steps",
    "original_steps",
    "original_dt",
}


def find_txt_file(txt_root: Path, output_name: str) -> Optional[Path]:
    direct = txt_root / output_name
    if direct.exists():
        return direct
    matches = list(txt_root.rglob(output_name))
    if matches:
        return matches[0]
    return None


def compute_sampled_dt(row: pd.Series, fallback_dt: Optional[float]) -> float:
    if fallback_dt is not None:
        return float(fallback_dt)

    original_dt = float(row["original_dt"])
    original_steps = float(row["original_steps"])
    sampled_steps = float(row["sampled_steps"])

    if not math.isfinite(original_dt) or original_dt <= 0:
        raise ValueError("original_dt가 유효하지 않습니다.")
    if sampled_steps <= 0 or original_steps <= 0:
        raise ValueError("sampled_steps / original_steps가 유효하지 않습니다.")

    return original_dt * (original_steps / sampled_steps)


def load_txt(txt_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        txt_path,
        sep=r"\s+",
        header=None,
        names=["frame_id", "track_id", "x", "y"],
        engine="python",
    )

    if df.empty:
        raise ValueError("TXT 파일이 비어 있습니다.")

    df = df.sort_values(["frame_id", "track_id"]).reset_index(drop=True)
    return df


def build_common_frames(df: pd.DataFrame) -> pd.DataFrame:
    track_ids = sorted(df["track_id"].unique().tolist())
    if len(track_ids) != 2:
        raise ValueError(f"track_id 개수가 2가 아닙니다. 실제 개수={len(track_ids)}")

    a = df[df["track_id"] == track_ids[0]][["frame_id", "x", "y"]].rename(
        columns={"x": "x1", "y": "y1"}
    )
    b = df[df["track_id"] == track_ids[1]][["frame_id", "x", "y"]].rename(
        columns={"x": "x2", "y": "y2"}
    )

    # 수정된 방법론: Outer join 후 결측치 선형 보간 처리 (충돌 직후 센서 끊김 방어)
    merged = a.merge(b, on="frame_id", how="outer").sort_values("frame_id").reset_index(drop=True)
    merged[["x1", "y1", "x2", "y2"]] = merged[["x1", "y1", "x2", "y2"]].interpolate(method="linear")
    merged = merged.dropna().reset_index(drop=True)

    if merged.empty:
        raise ValueError("보간 가능한 유효 공통 frame이 없습니다.")

    merged["distance"] = ((merged["x1"] - merged["x2"]) ** 2 + (merged["y1"] - merged["y2"]) ** 2) ** 0.5
    return merged


def extract_window_frames(
    merged: pd.DataFrame,
    sampled_dt: float,
    before_sec: float,
    after_sec: float,
    threshold_m: float,
    require_full_window: bool,
) -> Tuple[List[int], int, float]:
    near_mask = merged["distance"] <= threshold_m
    if not near_mask.any():
        raise RuntimeError("임계거리 이하가 되는 순간이 없습니다.")

    closest_idx = merged["distance"].idxmin()
    closest_frame = int(merged.loc[closest_idx, "frame_id"])
    min_distance = float(merged.loc[closest_idx, "distance"])

    frame_diffs = merged["frame_id"].diff().dropna()
    positive_diffs = frame_diffs[frame_diffs > 0]
    if positive_diffs.empty:
        frame_step = 10
    else:
        frame_step = int(round(float(positive_diffs.median())))
        if frame_step <= 0:
            frame_step = 10

    merged = merged.copy()
    merged["time_sec"] = ((merged["frame_id"] - merged["frame_id"].iloc[0]) / frame_step) * sampled_dt
    closest_time = float(merged.loc[closest_idx, "time_sec"])

    start_time = closest_time - before_sec
    end_time = closest_time + after_sec

    if require_full_window:
        if start_time < float(merged["time_sec"].iloc[0]) - 1e-9:
            raise RuntimeError("최소거리 시점 기준 이전 데이터가 부족합니다.")
        if end_time > float(merged["time_sec"].iloc[-1]) + 1e-9:
            raise RuntimeError("최소거리 시점 기준 이후 데이터가 부족합니다.")

    mask = (merged["time_sec"] >= start_time - 1e-9) & (merged["time_sec"] <= end_time + 1e-9)
    selected_frames = merged.loc[mask, "frame_id"].astype(int).tolist()
    if not selected_frames:
        raise RuntimeError("선택된 window에 해당하는 frame이 없습니다.")

    return selected_frames, closest_frame, min_distance


def remap_and_save(df: pd.DataFrame, selected_frames: List[int], save_path: Path) -> int:
    selected_frames_sorted = sorted(set(int(x) for x in selected_frames))
    frame0 = selected_frames_sorted[0]

    out_df = df[df["frame_id"].isin(selected_frames_sorted)].copy()
    if out_df.empty:
        raise RuntimeError("저장할 데이터가 없습니다.")

    out_df["frame_id"] = out_df["frame_id"].astype(int) - frame0
    out_df = out_df.sort_values(["frame_id", "track_id"]).reset_index(drop=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(save_path, sep="\t", header=False, index=False, float_format="%.6f")
    return len(out_df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSV를 참고해 2대 차량 TXT 중 최소거리 시점 기준 window만 추출합니다."
    )
    parser.add_argument("--csv", default="mat_preprocess/mat_txt/all/manifest.csv", help="참고 CSV 경로")
    parser.add_argument("--txt-root", default="mat_preprocess/mat_txt/all", help="원본 TXT 루트 폴더")
    parser.add_argument("--out-dir", default="mat_preprocess/mat_txt/collision_extract", help="결과 저장 폴더")
    parser.add_argument("--threshold-m", type=float, default=3.0, help="거리 임계값 n (meter)")
    parser.add_argument("--before-sec", type=float, default=4.0, help="최소거리 시점 이전 구간 (초)")
    parser.add_argument("--after-sec", type=float, default=0.5, help="최소거리 시점 이후 구간 (초)")
    # 정지 차량 판별을 위한 파라미터 추가
    parser.add_argument("--min-movement-m", type=float, default=2.0, help="차량의 최소 이동 거리 (초기값: 2.0m)")
    parser.add_argument(
        "--sampled-dt",
        type=float,
        default=None,
        help="TXT 한 step의 시간 간격(초). 지정하지 않으면 CSV의 original_dt/original_steps/sampled_steps로 계산",
    )
    parser.add_argument(
        "--allow-truncated-window",
        action="store_true",
        help="전/후 시간이 부족해도 가능한 범위만 저장",
    )
    parser.add_argument(
        "--output-suffix",
        default="_window",
        help="출력 파일명 suffix (기본: _window)",
    )
    parser.add_argument(
        "--summary-name",
        default="extraction_summary.csv",
        help="요약 CSV 파일명",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    txt_root = Path(args.txt_root)
    out_dir = Path(args.out_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(meta.columns)
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {sorted(missing)}")

    candidates = meta[(meta["status"] == "OK") & (meta["num_tracks"] == 2)].copy()
    if candidates.empty:
        print("처리할 대상이 없습니다. (status=OK, num_tracks=2 조건 결과 0개)")
        return

    results: List[Dict[str, object]] = []

    for _, row in tqdm(candidates.iterrows(), total=len(candidates), desc="충돌 추출 진행 중", unit="건"):
        output_name = str(row["output_name"])
        txt_path = find_txt_file(txt_root, output_name)
        record: Dict[str, object] = {
            "output_name": output_name,
            "txt_found": txt_path is not None,
            "status": "",
            "message": "",
            "saved_rows": 0,
            "closest_frame": "",
            "min_distance": "",
            "sampled_dt": "",
        }

        try:
            if txt_path is None:
                raise FileNotFoundError(f"TXT를 찾지 못했습니다: {output_name}")

            sampled_dt = compute_sampled_dt(row, args.sampled_dt)
            df = load_txt(txt_path)
            merged = build_common_frames(df)
            selected_frames, closest_frame, min_distance = extract_window_frames(
                merged=merged,
                sampled_dt=sampled_dt,
                before_sec=args.before_sec,
                after_sec=args.after_sec,
                threshold_m=args.threshold_m,
                require_full_window=not args.allow_truncated_window,
            )

            # --- 추가된 부분: 정지 차량(이동량이 매우 적은 차량) 필터링 로직 ---
            window_df = df[df["frame_id"].isin(selected_frames)]
            for track_id, group in window_df.groupby("track_id"):
                dx = group["x"].max() - group["x"].min()
                dy = group["y"].max() - group["y"].min()
                move_dist = (dx**2 + dy**2)**0.5
                
                if move_dist < args.min_movement_m:
                    raise RuntimeError(f"차량 {track_id}의 이동 거리({move_dist:.2f}m)가 임계값({args.min_movement_m}m) 미만입니다.")
            # -------------------------------------------------------------

            save_name = f"{Path(output_name).stem}{args.output_suffix}.txt"
            save_path = out_dir / save_name
            saved_rows = remap_and_save(df, selected_frames, save_path)

            record.update(
                {
                    "status": "SAVED",
                    "message": "",
                    "saved_rows": saved_rows,
                    "closest_frame": int(closest_frame),
                    "min_distance": float(min_distance),
                    "sampled_dt": float(sampled_dt),
                    "saved_path": str(save_path),
                }
            )
        except Exception as e:
            record.update({"status": "SKIPPED", "message": str(e)})

        results.append(record)

    summary_df = pd.DataFrame(results)
    
    out_dir.mkdir(parents=True, exist_ok=True) 
    summary_path = out_dir / args.summary_name
    summary_df.to_csv(summary_path, index=False)

    total = len(summary_df)
    saved = int((summary_df["status"] == "SAVED").sum())
    skipped = total - saved
    print(f"\n총 대상: {total}")
    print(f"저장 완료: {saved}")
    print(f"건너뜀: {skipped}")
    print(f"요약 CSV: {summary_path}")


if __name__ == "__main__":
    main()