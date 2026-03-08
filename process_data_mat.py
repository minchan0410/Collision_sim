import sys
import os
import numpy as np
import pandas as pd
import dill

# environment 모듈이 동일한 디렉토리 혹은 PYTHONPATH 내에 있어야 합니다.
from environment import Environment, Scene, Node, derivative_of

desired_max_time = 100
pred_indices = [2, 3]
state_dim = 6
frame_diff = 10
desired_frame_diff = 1
dt = 0.1

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    }
}

def maybe_makedirs(path_to_create):
    try:
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise

def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name)
    alpha = angle * np.pi / 180

    for node in scene.nodes:
        x = np.array(node.data.position.x, dtype=float)
        y = np.array(node.data.position.y, dtype=float)

        x, y = rotate_pc(np.array([x, y]), alpha)

        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        data_dict = {('position', 'x'): x,
                     ('position', 'y'): y,
                     ('velocity', 'x'): vx,
                     ('velocity', 'y'): vy,
                     ('acceleration', 'x'): ax,
                     ('acceleration', 'y'): ay}

        node_data = pd.DataFrame(data_dict, columns=data_columns)
        node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep)
        scene_aug.nodes.append(node)
        
    return scene_aug

def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    return scene_aug

# --- 메인 처리 로직 시작 ---

# 생성할 경로 지정
data_folder_name = 'processed_data'
maybe_makedirs(data_folder_name)

data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
base_input_dir = os.path.join('mat_preprocess', 'mat_txt')

for data_class in ['train', 'val', 'test']:
    target_dir = os.path.join(base_input_dir, data_class)
    
    # 해당 폴더가 없으면 건너뜀
    if not os.path.exists(target_dir):
        print(f"경고: {target_dir} 경로가 존재하지 않아 건너뜁니다.")
        continue
        
    print(f"[{data_class.upper()}] 데이터 처리를 시작합니다...")
    
    env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
    env.attention_radius = attention_radius

    scenes = []
    # 결과 파일명 설정 (예: process_data_mat/mat_train.pkl)
    data_dict_path = os.path.join(data_folder_name, f'mat_{data_class}.pkl')

    for subdir, dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.txt'):
                full_data_path = os.path.join(subdir, file)
                print('Processing:', full_data_path)

                # 수정 1: sep='\t' 대신 '\s+'를 사용하여 탭과 스페이스바 모두 유연하게 대응
                data = pd.read_csv(full_data_path, sep='\s+', index_col=False, header=None)
                
                # 데이터가 4열이라고 가정
                data = data.iloc[:, :4] 
                data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
                
                data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
                data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')

                # 수정 2: 프레임 다운샘플링 로직 (주의 파트 참고)
                # data['frame_id'] = data['frame_id'] // 10 

                data['frame_id'] -= data['frame_id'].min()

                data['node_type'] = 'PEDESTRIAN'
                data['node_id'] = data['track_id'].astype(str)

                data.sort_values('frame_id', inplace=True)

                # 위치 영점 조절 (Mean Centering)
                data['pos_x'] = data['pos_x'] - data['pos_x'].mean()
                data['pos_y'] = data['pos_y'] - data['pos_y'].mean()

                max_timesteps = data['frame_id'].max()

                # 씬 생성 (Train 일때만 Augmentation 함수 등록)
                scene = Scene(timesteps=max_timesteps+1, dt=dt, name=f"mat_{data_class}", aug_func=augment if data_class == 'train' else None)

                for node_id in pd.unique(data['node_id']):
                    node_df = data[data['node_id'] == node_id]
                    node_values = node_df[['pos_x', 'pos_y']].values

                    if node_values.shape[0] < 2:
                        continue

                    new_first_idx = node_df['frame_id'].iloc[0]

                    x = node_values[:, 0]
                    y = node_values[:, 1]
                    vx = derivative_of(x, scene.dt)
                    vy = derivative_of(y, scene.dt)
                    ax = derivative_of(vx, scene.dt)
                    ay = derivative_of(vy, scene.dt)

                    data_dict = {('position', 'x'): x,
                                 ('position', 'y'): y,
                                 ('velocity', 'x'): vx,
                                 ('velocity', 'y'): vy,
                                 ('acceleration', 'x'): ax,
                                 ('acceleration', 'y'): ay}

                    node_data = pd.DataFrame(data_dict, columns=data_columns)
                    node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data)
                    node.first_timestep = new_first_idx

                    scene.nodes.append(node)
                    
                if data_class == 'train':
                    scene.augmented = list()
                    angles = np.arange(0, 360, 15)
                    for angle in angles:
                        scene.augmented.append(augment_scene(scene, angle))

                scenes.append(scene)
                
    print(f'Processed {len(scenes)} scenes for data class {data_class}')
    env.scenes = scenes

    if len(scenes) > 0:
        with open(data_dict_path, 'wb') as f:
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
        print(f"Saved to {data_dict_path}\n")

print("모든 처리가 완료되었습니다.")