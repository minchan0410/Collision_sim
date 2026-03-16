from pathlib import Path
import yaml


def _cfg_get(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _pos_int(value, default):
    try:
        out = int(round(float(value)))
    except Exception:
        return int(default)
    return max(1, out)


def _pos_float(value, default):
    try:
        out = float(value)
    except Exception:
        return float(default)
    if out <= 0:
        return float(default)
    return out


def _load_mat_yaml_required():
    config_path = Path("configs") / "mat.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Required config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML content in {config_path}")
    return cfg


def get_traj_hypers(config=None):
    hypers = {
        'batch_size': 256,
        'grad_clip': 1.0,
        'learning_rate_style': 'exp',
        'min_learning_rate': 1e-05,
        'learning_decay_rate': 0.9999,
        'prediction_horizon': 20,
        'minimum_history_length': 12,
        'maximum_history_length': 12,
        'map_encoder':
            {'PEDESTRIAN':
                {'heading_state_index': 6,
                 'patch_size': [50, 10, 50, 90],
                 'map_channels': 3,
                 'hidden_channels': [10, 20, 10, 1],
                 'output_size': 32,
                 'masks': [5, 5, 5, 5],
                 'strides': [1, 1, 1, 1],
                 'dropout': 0.5
                 }
             },
        'k': 1,
        'k_eval': 25,
        'kl_min': 0.07,
        'kl_weight': 100.0,
        'kl_weight_start': 0,
        'kl_decay_rate': 0.99995,
        'kl_crossover': 400,
        'kl_sigmoid_divisor': 4,
        'rnn_kwargs':
            {'dropout_keep_prob': 0.75},
        'MLP_dropout_keep_prob': 0.9,
        'enc_rnn_dim_edge': 128,
        'enc_rnn_dim_edge_influence': 128,
        'enc_rnn_dim_history': 128,
        'enc_rnn_dim_future': 128,
        'dec_rnn_dim': 128,
        'q_z_xy_MLP_dims': None,
        'p_z_x_MLP_dims': 32,
        'GMM_components': 1,
        'log_p_yt_xz_max': 6,
        'N': 1,
        'tau_init': 2.0,
        'tau_final': 0.05,
        'tau_decay_rate': 0.997,
        'use_z_logit_clipping': True,
        'z_logit_clip_start': 0.05,
        'z_logit_clip_final': 5.0,
        'z_logit_clip_crossover': 300,
        'z_logit_clip_divisor': 5,
        'dynamic':
            {'PEDESTRIAN':
                {'name': 'SingleIntegrator',
                 'distribution': False,
                 'limits': {}
                 }
             },
        'state':
            {'PEDESTRIAN':
                {'position': ['x', 'y'],
                 'velocity': ['x', 'y'],
                 'acceleration': ['x', 'y'],
                 'heading': ['yaw', 'yaw_rate']
                 }
             },
        'pred_state': {'PEDESTRIAN': {'velocity': ['x', 'y']}},
        'log_histograms': False,
        'dynamic_edges': 'yes',
        'edge_state_combine_method': 'sum',
        'edge_influence_combine_method': 'attention',
        'edge_addition_filter': [0.25, 0.5, 0.75, 1.0],
        'edge_removal_filter': [1.0, 0.0],
        'offline_scene_graph': 'yes',
        'incl_robot_node': False,
        'node_freq_mult_train': False,
        'node_freq_mult_eval': False,
        'scene_freq_mult_train': False,
        'scene_freq_mult_eval': False,
        'scene_freq_mult_viz': False,
        'edge_encoding': True,
        'use_map_encoding': False,
        'augment': True,
        'override_attention_radius': [],
        'learning_rate': 0.01,
        'npl_rate': 0.8,
        'K': 80,
        'tao': 0.4
    }

    # Always use MAT timebase from configs/mat.yaml.
    mat_cfg = _load_mat_yaml_required()
    if "data_dt" not in mat_cfg:
        raise KeyError("'data_dt' is missing in configs/mat.yaml")
    data_dt = float(mat_cfg["data_dt"])
    if data_dt <= 0:
        raise ValueError(f"'data_dt' must be positive in configs/mat.yaml, got {data_dt}")
    hypers['dt'] = data_dt

    # 1) Explicit step counts override everything.
    ph_cfg = mat_cfg.get("prediction_horizon", None)
    min_h_cfg = mat_cfg.get("minimum_history_length", None)
    max_h_cfg = mat_cfg.get("maximum_history_length", None)
    if ph_cfg is not None:
        hypers['prediction_horizon'] = _pos_int(ph_cfg, hypers['prediction_horizon'])
    if min_h_cfg is not None:
        hypers['minimum_history_length'] = _pos_int(min_h_cfg, hypers['minimum_history_length'])
    if max_h_cfg is not None:
        hypers['maximum_history_length'] = _pos_int(max_h_cfg, hypers['maximum_history_length'])

    # 2) If step counts are not explicitly given, allow second-based config.
    if ph_cfg is None:
        pred_sec = mat_cfg.get("prediction_sec", None)
        if pred_sec is not None:
            hypers['prediction_horizon'] = _pos_int(float(pred_sec) / data_dt, hypers['prediction_horizon'])
        else:
            raise KeyError("Set either 'prediction_horizon' or 'prediction_sec' in configs/mat.yaml")

    if (min_h_cfg is None) and (max_h_cfg is None):
        history_sec = mat_cfg.get("history_sec", None)
        if history_sec is not None:
            h = _pos_int(float(history_sec) / data_dt, hypers['maximum_history_length'])
            hypers['minimum_history_length'] = h
            hypers['maximum_history_length'] = h
        else:
            min_hist_sec = mat_cfg.get("minimum_history_sec", None)
            max_hist_sec = mat_cfg.get("maximum_history_sec", None)
            if min_hist_sec is not None:
                hypers['minimum_history_length'] = _pos_int(float(min_hist_sec) / data_dt, hypers['minimum_history_length'])
            if max_hist_sec is not None:
                hypers['maximum_history_length'] = _pos_int(float(max_hist_sec) / data_dt, hypers['maximum_history_length'])
            if (min_hist_sec is None) and (max_hist_sec is None):
                raise KeyError(
                    "Set 'history_sec' or ('minimum_history_sec'/'maximum_history_sec') "
                    "or explicit history lengths in configs/mat.yaml"
                )

    if hypers['minimum_history_length'] > hypers['maximum_history_length']:
        hypers['minimum_history_length'] = hypers['maximum_history_length']

    return hypers
