import os
import argparse
import numpy as np
import pandas as pd
import yaml


# configs/mat.yaml에서 data_dt를 강제로 읽는다.
def load_data_dt_from_mat_yaml():
    config_path = os.path.join("configs", "mat.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Required config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML content in {config_path}")
    if "data_dt" not in cfg:
        raise KeyError(f"'data_dt' is missing in {config_path}")

    dt_cfg = float(cfg["data_dt"])
    if dt_cfg <= 0:
        raise ValueError(f"'data_dt' must be positive in {config_path}, got {dt_cfg}")
    return dt_cfg


parser = argparse.ArgumentParser(description="Estimate standardization stats from MAT TXT files.")
args = parser.parse_args()

dt = load_data_dt_from_mat_yaml()
if dt <= 0:
    raise ValueError(f"dt must be positive, got {dt}")
print(f"[INFO] getparam dt={dt:.6f} sec (from configs/mat.yaml)")


base_input_dir = os.path.join("mat_preprocess", "mat_txt")
folders = ["collision_extract_002"]

all_positions_x = []
all_positions_y = []
all_velocities_x = []
all_velocities_y = []
all_accelerations_x = []
all_accelerations_y = []
all_heading_yaw = []
all_heading_yaw_rate = []


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def build_yaw_series(raw_yaw, x, y, speed_eps=1.0e-3):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = x.shape[0]
    yaw = np.full((n,), np.nan, dtype=float)

    if raw_yaw is not None:
        raw = np.asarray(raw_yaw, dtype=float).reshape(-1)
        if raw.shape[0] == n:
            finite_raw = np.isfinite(raw)
            yaw[finite_raw] = wrap_to_pi(raw[finite_raw])

    if n >= 2:
        dx = np.diff(x)
        dy = np.diff(y)
        speed = np.hypot(dx, dy)
        motion = np.full((n,), np.nan, dtype=float)
        valid = np.isfinite(dx) & np.isfinite(dy) & (speed > speed_eps)
        motion[:-1][valid] = np.arctan2(dy[valid], dx[valid])
        motion[-1] = motion[-2]
        missing = ~np.isfinite(yaw)
        yaw[missing] = motion[missing]

    finite = np.isfinite(yaw)
    if not np.any(finite):
        return np.zeros((n,), dtype=float)

    first_valid = int(np.flatnonzero(finite)[0])
    yaw[:first_valid] = yaw[first_valid]
    prev = float(yaw[first_valid])
    for i in range(first_valid + 1, n):
        if np.isfinite(yaw[i]):
            prev = float(yaw[i])
        else:
            yaw[i] = prev
    return wrap_to_pi(yaw)


print("통계 계산을 시작합니다...")

for folder in folders:
    target_dir = os.path.join(base_input_dir, folder)
    if not os.path.exists(target_dir):
        continue

    for subdir, _, files in os.walk(target_dir):
        for file in files:
            if not file.endswith(".txt"):
                continue

            full_data_path = os.path.join(subdir, file)

            # TXT 데이터 로드
            data = pd.read_csv(full_data_path, sep=r"\s+", index_col=False, header=None)
            if data.shape[1] >= 5:
                data = data.iloc[:, :5]
                data.columns = ["frame_id", "track_id", "pos_x", "pos_y", "yaw"]
            else:
                data = data.iloc[:, :4]
                data.columns = ["frame_id", "track_id", "pos_x", "pos_y"]
                data["yaw"] = np.nan
            data["yaw"] = pd.to_numeric(data["yaw"], errors="coerce")

            # track_id, frame_id 기준으로 정렬
            data.sort_values(["track_id", "frame_id"], inplace=True)

            # process_data_mat.py와 동일하게 전체 평균 중심 정렬 기준 사용
            mean_x = data["pos_x"].mean()
            mean_y = data["pos_y"].mean()

            # track_id별 위치/속도/가속도/heading 통계 계산
            for _, group in data.groupby("track_id"):
                # 미분 안정성을 위해 최소 3포인트 필요
                if len(group) < 3:
                    continue

                pos_x = group["pos_x"].values
                pos_y = group["pos_y"].values

                # 위치 중심 정렬
                pos_x_centered = pos_x - mean_x
                pos_y_centered = pos_y - mean_y

                # 속도 계산 (v = dx/dt)
                vel_x = np.gradient(pos_x, dt)
                vel_y = np.gradient(pos_y, dt)

                # 가속도 계산 (a = dv/dt)
                acc_x = np.gradient(vel_x, dt)
                acc_y = np.gradient(vel_y, dt)

                # yaw / yaw_rate 계산
                yaw = build_yaw_series(group["yaw"].values, pos_x, pos_y)
                yaw_rate = np.gradient(np.unwrap(yaw), dt)

                all_positions_x.extend(pos_x_centered)
                all_positions_y.extend(pos_y_centered)
                all_velocities_x.extend(vel_x)
                all_velocities_y.extend(vel_y)
                all_accelerations_x.extend(acc_x)
                all_accelerations_y.extend(acc_y)
                all_heading_yaw.extend(yaw)
                all_heading_yaw_rate.extend(yaw_rate)


# 전체 데이터 기준 표준편차 계산
std_pos_x = np.std(all_positions_x) if all_positions_x else 1.0
std_pos_y = np.std(all_positions_y) if all_positions_y else 1.0
std_vel_x = np.std(all_velocities_x) if all_velocities_x else 2.0
std_vel_y = np.std(all_velocities_y) if all_velocities_y else 2.0
std_acc_x = np.std(all_accelerations_x) if all_accelerations_x else 1.0
std_acc_y = np.std(all_accelerations_y) if all_accelerations_y else 1.0
std_yaw = np.std(all_heading_yaw) if all_heading_yaw else float(np.pi)
std_yaw_rate = np.std(all_heading_yaw_rate) if all_heading_yaw_rate else 1.0

print("\n=== 계산 완료: process_data_mat.py의 standardization 값에 반영하세요 ===")
print("standardization = {")
print("    'PEDESTRIAN': {")
print("        'position': {")
print(f"            'x': {{'mean': 0, 'std': {std_pos_x:.2f}}},")
print(f"            'y': {{'mean': 0, 'std': {std_pos_y:.2f}}}")
print("        },")
print("        'velocity': {")
print(f"            'x': {{'mean': 0, 'std': {std_vel_x:.2f}}},")
print(f"            'y': {{'mean': 0, 'std': {std_vel_y:.2f}}}")
print("        },")
print("        'acceleration': {")
print(f"            'x': {{'mean': 0, 'std': {std_acc_x:.2f}}},")
print(f"            'y': {{'mean': 0, 'std': {std_acc_y:.2f}}}")
print("        },")
print("        'heading': {")
print(f"            'yaw': {{'mean': 0, 'std': {std_yaw:.2f}}},")
print(f"            'yaw_rate': {{'mean': 0, 'std': {std_yaw_rate:.2f}}}")
print("        }")
print("    }")
print("}")
