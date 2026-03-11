import os
import pandas as pd
import numpy as np

# 설정값 (process_data_mat.py와 동일하게 맞춤)
dt = 0.1
base_input_dir = os.path.join('mat_preprocess', 'mat_txt')
folders = ['collision_extract']

all_positions_x = []
all_positions_y = []
all_velocities_x = []
all_velocities_y = []
all_accelerations_x = []
all_accelerations_y = []

print("차량 데이터 스케일 분석을 시작합니다...")

for folder in folders:
    target_dir = os.path.join(base_input_dir, folder)
    if not os.path.exists(target_dir):
        continue

    for subdir, dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.txt'):
                full_data_path = os.path.join(subdir, file)
                
                # 데이터 로드
                data = pd.read_csv(full_data_path, sep='\s+', index_col=False, header=None)
                data = data.iloc[:, :4]
                data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
                
                # 시간 및 객체 순서대로 정렬
                data.sort_values(['track_id', 'frame_id'], inplace=True)
                
                # 파일 전체 위치 평균을 빼서 중앙 정렬 (process_data_mat.py 로직과 동일)
                mean_x = data['pos_x'].mean()
                mean_y = data['pos_y'].mean()
                
                # track_id 별로 속도 및 가속도 계산
                for track_id, group in data.groupby('track_id'):
                    if len(group) < 3: # 미분 계산을 위해 최소 3프레임 이상 필요
                        continue
                    
                    pos_x = group['pos_x'].values
                    pos_y = group['pos_y'].values
                    
                    # 중심화된 위치
                    pos_x_centered = pos_x - mean_x
                    pos_y_centered = pos_y - mean_y
                    
                    # 속도 계산 (v = dx/dt)
                    vel_x = np.gradient(pos_x, dt)
                    vel_y = np.gradient(pos_y, dt)
                    
                    # 가속도 계산 (a = dv/dt)
                    acc_x = np.gradient(vel_x, dt)
                    acc_y = np.gradient(vel_y, dt)
                    
                    all_positions_x.extend(pos_x_centered)
                    all_positions_y.extend(pos_y_centered)
                    all_velocities_x.extend(vel_x)
                    all_velocities_y.extend(vel_y)
                    all_accelerations_x.extend(acc_x)
                    all_accelerations_y.extend(acc_y)

# 전체 데이터에 대한 표준편차 계산
std_pos_x = np.std(all_positions_x) if all_positions_x else 1.0
std_pos_y = np.std(all_positions_y) if all_positions_y else 1.0
std_vel_x = np.std(all_velocities_x) if all_velocities_x else 2.0
std_vel_y = np.std(all_velocities_y) if all_velocities_y else 2.0
std_acc_x = np.std(all_accelerations_x) if all_accelerations_x else 1.0
std_acc_y = np.std(all_accelerations_y) if all_accelerations_y else 1.0

print("\n=== 계산 완료! process_data_mat.py의 standardization 부분을 아래 코드로 교체하세요 ===")
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
print("        }")
print("    }")
print("}")