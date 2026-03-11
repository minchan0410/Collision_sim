import sys
import os
import argparse
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
            'x': {'mean': 0, 'std': 6.22},
            'y': {'mean': 0, 'std': 27.19}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 4.20},
            'y': {'mean': 0, 'std': 9.82}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1.88},
            'y': {'mean': 0, 'std': 2.68}
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

# def augment(scene):
#     # scene_aug = np.random.choice(scene.augmented)
#     # scene_aug.temporal_scene_graph = scene.temporal_scene_graph
#     # return scene_aug
#     return scene

def augment(scene):
    # 원본 씬과 미세 회전된 씬들을 합쳐서 그 중 하나를 무작위로 선택
    choices = [scene] + (scene.augmented if hasattr(scene, 'augmented') else [])
    scene_aug = np.random.choice(choices)
    
    # 모델 학습에 필수적인 씬 그래프(주변 차량 관계망) 정보를 복사해줌
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    return scene_aug


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="분리된 보행자 데이터셋을 전처리합니다.")
    parser.add_argument('--suffix', type=str, default='', help='데이터셋 폴더의 접미사 (예: mimi 입력 시 train_mimi 탐색)')
    args = parser.parse_args()

    data_folder_name = 'processed_data'
    maybe_makedirs(data_folder_name)

    data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
    base_input_dir = os.path.join('mat_preprocess', 'mat_txt')

    dir_suffix = f"_{args.suffix}" if args.suffix else ""

    for base_class in ['train', 'val', 'test']:
        # 읽어올 대상 폴더명 (예: train_mimi)
        folder_name = f"{base_class}{dir_suffix}"
        target_dir = os.path.join(base_input_dir, folder_name)
        
        # [수정됨] 저장될 파일 및 Scene 객체의 이름 설정 (예: mat_mimi_train)
        if args.suffix:
            output_name = f"mat_{args.suffix}_{base_class}"
        else:
            output_name = f"mat_{base_class}"
        
        if not os.path.exists(target_dir):
            print(f"경고: {target_dir} 경로가 존재하지 않아 건너뜁니다.")
            continue
            
        print(f"[{folder_name.upper()}] 데이터 처리를 시작합니다...")
        
        env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 50
        env.attention_radius = attention_radius

        scenes = []
        
        # [수정됨] 파일명에 output_name 적용
        data_dict_path = os.path.join(data_folder_name, f'{output_name}.pkl')

        for subdir, dirs, files in os.walk(target_dir):
            for file in files:
                if file.endswith('.txt'):
                    full_data_path = os.path.join(subdir, file)
                    print('Processing:', full_data_path)

                    data = pd.read_csv(full_data_path, sep='\s+', index_col=False, header=None)
                    
                    data = data.iloc[:, :4] 
                    data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
                    
                    data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
                    data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')

                    data['frame_id'] -= data['frame_id'].min()

                    data['node_type'] = 'PEDESTRIAN'
                    data['node_id'] = data['track_id'].astype(str)

                    data.sort_values('frame_id', inplace=True)

                    data['pos_x'] = data['pos_x'] - data['pos_x'].mean()
                    data['pos_y'] = data['pos_y'] - data['pos_y'].mean()

                    max_timesteps = data['frame_id'].max()

                    # [수정됨] Scene 생성 시에도 name에 output_name을 할당하여 내부 데이터 구조의 일관성 유지
                    scene = Scene(timesteps=max_timesteps+1, dt=dt, name=output_name, aug_func=augment if base_class == 'train' else None)

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
                        
                    if base_class == 'train':
                    #     pass
                    #     # scene.augmented = list()
                    #     # angles = np.arange(0, 360, 15)
                    #     # for angle in angles:
                    #     #     scene.augmented.append(augment_scene(scene, angle))
                        scene.augmented = list()
                        num_aug = 4
                        angles = np.random.uniform(-3.0, 3.0, size=num_aug)
                        
                        for angle in angles:
                            scene.augmented.append(augment_scene(scene, angle))

                    scenes.append(scene)
                    
        print(f'Processed {len(scenes)} scenes for data class {folder_name} (Saved as {output_name})')
        env.scenes = scenes

        if len(scenes) > 0:
            with open(data_dict_path, 'wb') as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
            print(f"Saved to {data_dict_path}\n")

    print("모든 처리가 완료되었습니다.")