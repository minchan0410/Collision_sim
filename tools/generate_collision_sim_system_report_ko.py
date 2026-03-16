from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import runpy
import yaml
from matplotlib import font_manager, rcParams
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "mat.yaml"
OUT_DIR = ROOT / "output"
ASSET_DIR = OUT_DIR / "report_assets"
PDF_PATH = OUT_DIR / "collision_sim_system_report_ko.pdf"


def setup_korean_font() -> str:
    candidates = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for p in candidates:
        if p.exists():
            font_manager.fontManager.addfont(str(p))
            family = font_manager.FontProperties(fname=str(p)).get_name()
            rcParams["font.family"] = family
            rcParams["axes.unicode_minus"] = False
            return family
    rcParams["axes.unicode_minus"] = False
    return "sans-serif"


def load_config() -> Dict:
    if not CFG_PATH.exists():
        return {}
    with CFG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def add_box(ax, x, y, w, h, text, fc="#f4f6f8", ec="#2f3b52", fontsize=10):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def add_arrow(ax, x1, y1, x2, y2, color="#1f2a44"):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, linewidth=1.5, color=color)
    ax.add_patch(arr)


def page_cover(pdf: PdfPages, cfg: Dict, hypers: Dict):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    ax.text(0.06, 0.95, "Collision_sim 모델 구조/동역학 가이드 종합 문서", fontsize=21, fontweight="bold", va="top")
    ax.text(0.06, 0.915, f"생성 시각: {now}", fontsize=11, color="dimgray")
    ax.text(0.06, 0.885, "코드 기준: mid.py, models/autoencoder.py, models/diffusion.py, utils/trajectron_hypers.py", fontsize=10)

    lines = [
        f"- 데이터셋: {cfg.get('dataset', 'mat_collision')}  |  샘플링: {cfg.get('sampling', 'ddpm')}",
        f"- 입력 상태(state): {hypers['state']['PEDESTRIAN']}",
        f"- 예측 타깃(pred_state): {hypers['pred_state']['PEDESTRIAN']}",
        f"- 히스토리/예측 길이: {hypers['maximum_history_length']} / {hypers['prediction_horizon']}",
        f"- Dynamics Guidance: {cfg.get('dynamics_guidance_enabled', True)}",
        f"- Bicycle Rollout Projection: {cfg.get('bicycle_rollout_enabled', True)}",
        f"- Yaw 보조 loss: {cfg.get('yaw_loss_enabled', True)} (weight={cfg.get('yaw_loss_weight', 0.1)})",
    ]
    y = 0.82
    for ln in lines:
        ax.text(0.08, y, ln, fontsize=12)
        y -= 0.04

    ax.text(
        0.06,
        0.12,
        "이 문서는 모델의 입력/출력 구조, 학습/생성 경로, 동역학 제약의 적용 위치 및 수식을\n"
        "시각적으로 한 번에 파악할 수 있도록 구성했습니다.",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef6ff", edgecolor="#9ab6d8"),
    )
    pdf.savefig(fig)
    plt.close(fig)


def page_architecture(pdf: PdfPages):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.06, 0.96, "1) 모델 구조 시각화 (학습/생성)", fontsize=17, fontweight="bold", va="top")

    # Training flow
    ax.text(0.07, 0.90, "학습 경로", fontsize=13, fontweight="bold", color="#253a73")
    add_box(ax, 0.07, 0.81, 0.19, 0.07, "Batch\n(x_t, y_st_t, ...)", fc="#f1f7ff")
    add_box(ax, 0.31, 0.81, 0.20, 0.07, "Trajectron Encoder\nget_latent()", fc="#f1f7ff")
    add_box(ax, 0.56, 0.81, 0.18, 0.07, "Diffusion Loss\nMSE(eps)", fc="#fff7ec")
    add_box(ax, 0.77, 0.81, 0.16, 0.07, "Yaw Aux Loss\n(옵션)", fc="#fff7ec")
    add_arrow(ax, 0.26, 0.845, 0.31, 0.845)
    add_arrow(ax, 0.51, 0.845, 0.56, 0.845)
    add_arrow(ax, 0.74, 0.845, 0.77, 0.845)
    ax.text(0.63, 0.775, "L_total = L_diff + λ_yaw·L_yaw", fontsize=11, color="#8a4d00")

    # Inference flow
    ax.text(0.07, 0.66, "생성(샘플링) 경로", fontsize=13, fontweight="bold", color="#253a73")
    add_box(ax, 0.07, 0.57, 0.19, 0.075, "Context z\n(encoder output)", fc="#edf8f3")
    add_box(ax, 0.30, 0.57, 0.21, 0.075, "Diffusion.sample\n(velocity_st 생성)", fc="#edf8f3")
    add_box(ax, 0.55, 0.57, 0.17, 0.075, "Destandardize\nv_phys", fc="#edf8f3")
    add_box(ax, 0.75, 0.57, 0.18, 0.075, "SingleIntegrator\n기본 위치 적분", fc="#edf8f3")
    add_arrow(ax, 0.26, 0.607, 0.30, 0.607)
    add_arrow(ax, 0.51, 0.607, 0.55, 0.607)
    add_arrow(ax, 0.72, 0.607, 0.75, 0.607)

    add_box(ax, 0.16, 0.44, 0.32, 0.08, "Dynamics Guidance\nx <- x - λ∇G(x)\n(denoising 중간 스텝마다)", fc="#ffeef0", ec="#9e2a2b")
    add_box(ax, 0.56, 0.44, 0.33, 0.08, "Bicycle Rollout Projection\n(최종 속도/궤적을 차량 기하 제약으로 보정)", fc="#eef3ff", ec="#375a9e")
    add_arrow(ax, 0.40, 0.57, 0.32, 0.52, color="#9e2a2b")
    add_arrow(ax, 0.84, 0.57, 0.72, 0.52, color="#375a9e")

    add_box(ax, 0.25, 0.31, 0.50, 0.08, "최종 출력: position, velocity, speed, yaw", fc="#f7f7f7", ec="#374151", fontsize=12)
    add_arrow(ax, 0.36, 0.44, 0.46, 0.39)
    add_arrow(ax, 0.72, 0.44, 0.56, 0.39)

    ax.text(
        0.07,
        0.17,
        "핵심: 현재 구조는 '속도 예측 모델 + 제약 기반 샘플 보정'이며,\n"
        "하드 제약(절대 금지)이 아니라 소프트 패널티 기반 유도 방식입니다.",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffceb", edgecolor="#d2b55b"),
    )
    pdf.savefig(fig)
    plt.close(fig)

    # Export as image too
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig2 = plt.figure(figsize=(14, 8))
    ax2 = fig2.add_axes([0, 0, 1, 1])
    ax2.axis("off")
    ax2.text(0.03, 0.95, "Collision_sim 구조 요약", fontsize=20, fontweight="bold")
    add_box(ax2, 0.04, 0.72, 0.24, 0.12, "Trajectron Encoder")
    add_box(ax2, 0.35, 0.72, 0.24, 0.12, "Diffusion (vel)")
    add_box(ax2, 0.66, 0.72, 0.28, 0.12, "Guidance + Bicycle Projection")
    add_arrow(ax2, 0.28, 0.78, 0.35, 0.78)
    add_arrow(ax2, 0.59, 0.78, 0.66, 0.78)
    add_box(ax2, 0.24, 0.45, 0.52, 0.14, "Output: position / velocity / speed / yaw")
    add_arrow(ax2, 0.50, 0.72, 0.50, 0.59)
    fig2.savefig(ASSET_DIR / "architecture_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig2)


def page_io_and_formulas(pdf: PdfPages, cfg: Dict, hypers: Dict):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.06, 0.96, "2) 입력/출력 구조와 학습 수식", fontsize=17, fontweight="bold", va="top")

    state = hypers["state"]["PEDESTRIAN"]
    pred_state = hypers["pred_state"]["PEDESTRIAN"]
    lines = [
        f"입력 state: {state}",
        f"예측 타깃 pred_state: {pred_state}",
        "",
        "학습 배치 주요 텐서:",
        "  - x_t: 히스토리 상태 (표준화/비표준화 버전 포함)",
        "  - y_st_t: 미래 타깃 (현재는 velocity x,y)",
        "  - context: Trajectron encoder latent",
        "",
        "Diffusion 손실:",
        "  x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * eps",
        "  L_diff = MSE(eps_theta(x_t, t, context), eps)",
        "",
        "Yaw 보조 손실(옵션):",
        "  v_hat = x0_hat * std,  v_gt = y_st_t * std",
        "  L_yaw = mean[1 - cos(psi_hat - psi_gt)]  (저속 마스크 적용)",
        f"  L_total = L_diff + {cfg.get('yaw_loss_weight', 0.1)} * L_yaw",
    ]
    y = 0.90
    for ln in lines:
        ax.text(0.08, y, ln, fontsize=11)
        y -= 0.035

    ax.text(
        0.08,
        0.32,
        "학습에서 중요한 점:\n"
        "1) Dynamics Guidance는 주로 샘플링 보정(inference-time correction)\n"
        "2) Yaw 오차를 학습에 직접 반영하는 것은 yaw_aux_loss",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef6ff", edgecolor="#8fb0d8"),
    )
    pdf.savefig(fig)
    plt.close(fig)


def page_guidance(pdf: PdfPages, cfg: Dict):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.07, 0.06, 0.86, 0.88])
    ax.axis("off")
    ax.text(0.0, 1.0, "3) Dynamics Guidance 적용 위치/의미", fontsize=17, fontweight="bold", va="top")

    txt = [
        "적용 위치: diffusion denoising 루프의 각 step에서 x_{t-1} 계산 후 보정",
        "업데이트: x <- x - lambda * grad G(x)",
        "",
        "G(x) 구성 항(soft constraint):",
        "- accel, jerk, yaw_rate, curvature, lateral_accel, slip_ratio, reverse motion",
        "- 각 항은 relu(위반량)^2 형태로 누적",
        "",
        "중요: 하드 제약이 아니므로 위반을 '완전히 0'으로 보장하지는 않음",
        "대신 위반 방향의 gradient로 샘플을 지속적으로 밀어내는 방식",
        "",
        f"현재 핵심 설정: scale={cfg.get('dynamics_guidance_scale', 0.03)}, "
        f"inner_steps={cfg.get('dynamics_guidance_inner_steps', 1)}, "
        f"max_grad_norm={cfg.get('dynamics_guidance_max_grad_norm', 1.5)}",
    ]
    yy = 0.92
    for line in txt:
        ax.text(0.02, yy, line, fontsize=11)
        yy -= 0.06 if line == "" else 0.045

    # Simple visual: penalty weights bar
    ax2 = fig.add_axes([0.15, 0.08, 0.72, 0.28])
    names = ["accel", "jerk", "yaw", "curv", "lat", "slip", "rev"]
    values = [
        cfg.get("dynamics_guidance_weight_accel", 1.0),
        cfg.get("dynamics_guidance_weight_jerk", 0.6),
        cfg.get("dynamics_guidance_weight_yaw_rate", 0.8),
        cfg.get("dynamics_guidance_weight_curvature", 1.0),
        cfg.get("dynamics_guidance_weight_lateral_accel", 0.9),
        cfg.get("dynamics_guidance_weight_slip", 0.4),
        cfg.get("dynamics_guidance_weight_reverse", 0.5),
    ]
    ax2.bar(names, values, color=["#325d88", "#4c78a8", "#5f9ed1", "#7bb7e0", "#95c8ec", "#b7daf5", "#d6ecff"])
    ax2.set_ylabel("weight")
    ax2.set_title("Guidance 항목별 가중치 (mat.yaml)")
    ax2.grid(axis="y", alpha=0.3)

    pdf.savefig(fig)
    plt.close(fig)


def page_bicycle(pdf: PdfPages, cfg: Dict):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.06, 0.96, "4) Bicycle 기반 경로 보정 (Rollout Projection)", fontsize=17, fontweight="bold", va="top")

    wheelbase = float(cfg.get("bicycle_rollout_wheelbase", 2.7))
    steer_deg = float(cfg.get("bicycle_rollout_max_steer_deg", 32.0))
    steer_rad = abs(steer_deg) * math.pi / 180.0
    kappa_max = abs(math.tan(steer_rad)) / max(wheelbase, 1e-6)
    yaw_rate_max = float(cfg.get("dynamics_guidance_max_yaw_rate", 0.6))
    low_th = float(cfg.get("bicycle_rollout_low_speed_threshold", 0.3))

    lines = [
        "목적: 저속/정지 근처의 비현실적 제자리 회전을 줄이기 위해 생성 결과를 기하학적으로 보정",
        "",
        "핵심 수식:",
        "  kappa_max = tan(delta_max) / wheelbase",
        "  yaw_rate_lim(v) = min(yaw_rate_cfg, v * kappa_max)",
        "  delta_psi <- clamp(delta_psi, -yaw_rate_lim*dt, +yaw_rate_lim*dt)",
        "  v_x = v*cos(psi), v_y = v*sin(psi), p_{t+1} = p_t + v_t*dt",
        "",
        f"현재 값: wheelbase={wheelbase:.2f}m, max_steer_deg={steer_deg:.1f}deg, "
        f"kappa_max={kappa_max:.3f} 1/m, yaw_rate_cfg={yaw_rate_max:.2f}rad/s",
        f"저속 yaw 고정 threshold={low_th:.2f}m/s",
    ]
    y = 0.88
    for ln in lines:
        ax.text(0.08, y, ln, fontsize=11)
        y -= 0.048 if ln == "" else 0.038

    # Plot yaw_rate limit curve
    axp = fig.add_axes([0.12, 0.22, 0.76, 0.28])
    speeds = [i * 0.2 for i in range(0, 61)]  # 0..12
    dyn_lim = [s * kappa_max for s in speeds]
    eff_lim = [min(yaw_rate_max, d) for d in dyn_lim]
    axp.plot(speeds, dyn_lim, "--", color="#5a88c9", label="v * kappa_max")
    axp.plot(speeds, eff_lim, "-", color="#1f3a75", linewidth=2, label="effective yaw_rate limit")
    axp.axvline(low_th, color="#c0392b", linestyle=":", label="low-speed threshold")
    axp.set_xlabel("speed (m/s)")
    axp.set_ylabel("yaw_rate limit (rad/s)")
    axp.set_title("속도별 허용 yaw_rate (bicycle 기반)")
    axp.grid(alpha=0.3)
    axp.legend(loc="lower right")

    pdf.savefig(fig)
    plt.close(fig)

    # Export image too
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig if False else None  # keep linter quiet in standalone style
    fig2 = plt.figure(figsize=(10, 5))
    ax2 = fig2.add_subplot(111)
    ax2.plot(speeds, eff_lim, color="#1f3a75", linewidth=2)
    ax2.set_title("Bicycle yaw-rate limit curve")
    ax2.set_xlabel("speed (m/s)")
    ax2.set_ylabel("yaw_rate limit (rad/s)")
    ax2.grid(alpha=0.3)
    fig2.savefig(ASSET_DIR / "bicycle_yaw_limit_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig2)


def page_config_matrix(pdf: PdfPages, cfg: Dict):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.06, 0.96, "5) 현재 핵심 설정값 요약 (mat.yaml)", fontsize=17, fontweight="bold", va="top")

    keys = [
        "sampling",
        "dynamics_guidance_enabled",
        "dynamics_guidance_scale",
        "dynamics_guidance_min_speed",
        "dynamics_guidance_max_yaw_rate",
        "dynamics_guidance_max_curvature",
        "dynamics_guidance_use_bicycle_curvature",
        "dynamics_guidance_wheelbase",
        "dynamics_guidance_max_steer_deg",
        "dynamics_guidance_low_speed_yaw_weight",
        "dynamics_guidance_low_speed_yaw_threshold",
        "yaw_loss_enabled",
        "yaw_loss_weight",
        "bicycle_rollout_enabled",
        "bicycle_rollout_wheelbase",
        "bicycle_rollout_max_steer_deg",
        "bicycle_rollout_low_speed_threshold",
        "viz_vehicle_boxes_enabled",
    ]

    rows = [["key", "value"]]
    for k in keys:
        rows.append([k, str(cfg.get(k, "<not set>"))])

    tbl_ax = fig.add_axes([0.08, 0.08, 0.84, 0.82])
    tbl_ax.axis("off")
    table = tbl_ax.table(cellText=rows, loc="upper left", cellLoc="left", colWidths=[0.58, 0.30])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.45)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#93a1b3")
        if r == 0:
            cell.set_facecolor("#e9eef5")
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f8fafc")

    pdf.savefig(fig)
    plt.close(fig)


def page_code_map(pdf: PdfPages):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.06, 0.96, "6) 코드 맵 (어디를 보면 되는지)", fontsize=17, fontweight="bold", va="top")

    lines = [
        "학습 루프",
        "  - mid.py: MID.train()",
        "",
        "시각화(epoch)",
        "  - mid.py: _visualize_epoch(), _draw_vehicle_rectangles()",
        "",
        "잠재 인코딩/생성/손실",
        "  - models/autoencoder.py: encode(), generate(), get_loss()",
        "  - models/autoencoder.py: _build_dynamics_guidance(), _rollout_bicycle_projection()",
        "",
        "Diffusion + Guidance 본체",
        "  - models/diffusion.py: get_loss(), sample()",
        "  - models/diffusion.py: _compute_dynamics_objective(), _apply_dynamics_guidance()",
        "",
        "입력/출력 state 정의",
        "  - utils/trajectron_hypers.py: state / pred_state",
        "",
        "데이터 전처리(yaw 포함)",
        "  - process_data_mat.py: build_yaw_series(), yaw/yaw_rate 계산",
    ]
    y = 0.90
    for ln in lines:
        ax.text(0.09, y, ln, fontsize=11)
        y -= 0.045 if ln else 0.02

    ax.text(
        0.09,
        0.14,
        "참고: 보고서에 포함된 구조도 PNG는 output/report_assets/ 아래에도 함께 저장됩니다.",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fffceb", edgecolor="#d4b75a"),
    )
    pdf.savefig(fig)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    setup_korean_font()

    cfg = load_config()
    hypers_mod = runpy.run_path(str(ROOT / "utils" / "trajectron_hypers.py"))
    hypers = hypers_mod["get_traj_hypers"]()

    with PdfPages(PDF_PATH) as pdf:
        page_cover(pdf, cfg, hypers)
        page_architecture(pdf)
        page_io_and_formulas(pdf, cfg, hypers)
        page_guidance(pdf, cfg)
        page_bicycle(pdf, cfg)
        page_config_matrix(pdf, cfg)
        page_code_map(pdf)

    print(f"saved: {PDF_PATH}")
    print(f"assets: {ASSET_DIR}")


if __name__ == "__main__":
    main()
