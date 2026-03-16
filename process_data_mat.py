import sys
import os
import argparse
import numpy as np
import pandas as pd
import dill
import yaml

# environment 紐⑤뱢???숈씪???붾젆?좊━ ?뱀? PYTHONPATH ?댁뿉 ?덉뼱???⑸땲??
from environment import Environment, Scene, Node, derivative_of

desired_max_time = 100
pred_indices = [2, 3]
state_dim = 8
frame_diff = 10
desired_frame_diff = 1
dt = None

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 3.91},
            'y': {'mean': 0, 'std': 13.49}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 5.06},
            'y': {'mean': 0, 'std': 9.61}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 3.93},
            'y': {'mean': 0, 'std': 3.30}
        },
        'heading': {
            'yaw': {'mean': 0, 'std': 0.57},
            'yaw_rate': {'mean': 0, 'std': 0.14}
        }
    }
}


def make_data_columns():
    return pd.MultiIndex.from_tuples([
        ('position', 'x'),
        ('position', 'y'),
        ('velocity', 'x'),
        ('velocity', 'y'),
        ('acceleration', 'x'),
        ('acceleration', 'y'),
        ('heading', 'yaw'),
        ('heading', 'yaw_rate'),
    ])


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def build_yaw_series(raw_yaw, x, y, speed_eps=1.0e-3):
    """
    Build a finite yaw(rad) sequence from:
    1) raw yaw column if present,
    2) fallback heading from trajectory direction.
    """
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

def maybe_makedirs(path_to_create):
    try:
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise


def load_data_dt_from_mat_yaml():
    config_path = os.path.join('configs', 'mat.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Required config not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML content in {config_path}")
    if 'data_dt' not in cfg:
        raise KeyError(f"'data_dt' is missing in {config_path}")

    dt_cfg = float(cfg['data_dt'])
    if dt_cfg <= 0:
        raise ValueError(f"'data_dt' must be positive in {config_path}, got {dt_cfg}")
    return dt_cfg

def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    data_columns = make_data_columns()
    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name)
    alpha = angle * np.pi / 180

    for node in scene.nodes:
        x_src = np.array(node.data.position.x, dtype=float)
        y_src = np.array(node.data.position.y, dtype=float)

        x, y = rotate_pc(np.array([x_src, y_src]), alpha)

        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        try:
            yaw_src = np.array(node.data[:, ('heading', 'yaw')], dtype=float).reshape(-1)
        except Exception:
            yaw_src = build_yaw_series(None, x_src, y_src)
        yaw = wrap_to_pi(yaw_src + alpha)
        yaw_rate = derivative_of(np.unwrap(yaw), scene.dt)

        data_dict = {('position', 'x'): x,
                     ('position', 'y'): y,
                     ('velocity', 'x'): vx,
                     ('velocity', 'y'): vy,
                     ('acceleration', 'x'): ax,
                     ('acceleration', 'y'): ay,
                     ('heading', 'yaw'): yaw,
                     ('heading', 'yaw_rate'): yaw_rate}

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
    # ?먮낯 ?ш낵 誘몄꽭 ?뚯쟾???щ뱾???⑹퀜??洹?以??섎굹瑜?臾댁옉?꾨줈 ?좏깮
    choices = [scene] + (scene.augmented if hasattr(scene, 'augmented') else [])
    scene_aug = np.random.choice(choices)
    
    # 紐⑤뜽 ?숈뒿???꾩닔?곸씤 ??洹몃옒??二쇰? 李⑤웾 愿怨꾨쭩) ?뺣낫瑜?蹂듭궗?댁쨲
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    return scene_aug


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="遺꾨━??蹂댄뻾???곗씠?곗뀑???꾩쿂由ы빀?덈떎.")
    parser.add_argument('--suffix', type=str, default='', help='?곗씠?곗뀑 ?대뜑???묐???(?? mimi ?낅젰 ??train_mimi ?먯깋)')
    args = parser.parse_args()

    dt = load_data_dt_from_mat_yaml()
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    print(f"[INFO] process_data_mat dt={dt:.6f} sec (from configs/mat.yaml)")

    data_folder_name = 'processed_data'
    maybe_makedirs(data_folder_name)

    data_columns = make_data_columns()
    base_input_dir = os.path.join('mat_preprocess', 'mat_txt')

    dir_suffix = f"_{args.suffix}" if args.suffix else ""

    for base_class in ['train', 'val', 'test']:
        # ?쎌뼱??????대뜑紐?(?? train_mimi)
        folder_name = f"{base_class}{dir_suffix}"
        target_dir = os.path.join(base_input_dir, folder_name)
        
        # [?섏젙?? ??λ맆 ?뚯씪 諛?Scene 媛앹껜???대쫫 ?ㅼ젙 (?? mat_mimi_train)
        if args.suffix:
            output_name = f"mat_{args.suffix}_{base_class}"
        else:
            output_name = f"mat_{base_class}"
        
        if not os.path.exists(target_dir):
            print(f"寃쎄퀬: {target_dir} 寃쎈줈媛 議댁옱?섏? ?딆븘 嫄대꼫?곷땲??")
            continue
            
        print(f"[{folder_name.upper()}] ?곗씠??泥섎━瑜??쒖옉?⑸땲??..")
        
        env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 50
        env.attention_radius = attention_radius

        scenes = []
        
        # [?섏젙?? ?뚯씪紐낆뿉 output_name ?곸슜
        data_dict_path = os.path.join(data_folder_name, f'{output_name}.pkl')

        for subdir, dirs, files in os.walk(target_dir):
            for file in files:
                if file.endswith('.txt'):
                    full_data_path = os.path.join(subdir, file)
                    print('Processing:', full_data_path)

                    data = pd.read_csv(full_data_path, sep=r'\s+', index_col=False, header=None)

                    if data.shape[1] >= 5:
                        data = data.iloc[:, :5]
                        data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y', 'yaw']
                    else:
                        data = data.iloc[:, :4]
                        data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
                        data['yaw'] = np.nan
                    
                    data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
                    data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')
                    data['yaw'] = pd.to_numeric(data['yaw'], errors='coerce')

                    data['frame_id'] -= data['frame_id'].min()

                    data['node_type'] = 'PEDESTRIAN'
                    data['node_id'] = data['track_id'].astype(str)

                    data.sort_values('frame_id', inplace=True)

                    data['pos_x'] = data['pos_x'] - data['pos_x'].mean()
                    data['pos_y'] = data['pos_y'] - data['pos_y'].mean()

                    max_timesteps = data['frame_id'].max()

                    # [?섏젙?? Scene ?앹꽦 ?쒖뿉??name??output_name???좊떦?섏뿬 ?대? ?곗씠??援ъ“???쇨????좎?
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
                        yaw_raw = node_df['yaw'].to_numpy(dtype=float)
                        yaw = build_yaw_series(yaw_raw, x, y)
                        yaw_rate = derivative_of(np.unwrap(yaw), scene.dt)

                        data_dict = {('position', 'x'): x,
                                     ('position', 'y'): y,
                                     ('velocity', 'x'): vx,
                                     ('velocity', 'y'): vy,
                                     ('acceleration', 'x'): ax,
                                     ('acceleration', 'y'): ay,
                                     ('heading', 'yaw'): yaw,
                                     ('heading', 'yaw_rate'): yaw_rate}

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
                        angles = np.random.uniform(-180.0, 180.0, size=num_aug)
                        
                        for angle in angles:
                            scene.augmented.append(augment_scene(scene, angle))

                    scenes.append(scene)
                    
        print(f'Processed {len(scenes)} scenes for data class {folder_name} (Saved as {output_name})')
        env.scenes = scenes

        if len(scenes) > 0:
            with open(data_dict_path, 'wb') as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
            print(f"Saved to {data_dict_path}\n")

    print("紐⑤뱺 泥섎━媛 ?꾨즺?섏뿀?듬땲??")
