import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "mat.yaml"
OUT_PATH = ROOT / "output" / "collision_sim_model_guidance_analysis_ko.pdf"


def parse_simple_yaml(path: Path):
    cfg = {}
    if not path.exists():
        return cfg

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue

        lower = value.lower()
        if lower in {"true", "false"}:
            cfg[key] = lower == "true"
            continue

        try:
            if "." in value or "e" in lower:
                cfg[key] = float(value)
            else:
                cfg[key] = int(value)
            continue
        except Exception:
            pass

        cfg[key] = value

    return cfg


def setup_korean_font():
    candidates = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]

    for font_path in candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            family_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            rcParams["font.family"] = family_name
            rcParams["axes.unicode_minus"] = False
            return family_name

    rcParams["axes.unicode_minus"] = False
    return "sans-serif"


def wrap_lines(lines, width=74):
    out = []
    for line in lines:
        if not line:
            out.append("")
            continue
        if line.startswith("    "):
            out.append(line)
            continue
        wrapped = textwrap.wrap(line, width=width, break_long_words=False, break_on_hyphens=False)
        out.extend(wrapped if wrapped else [""])
    return out


def draw_page(pdf, title, lines, page_no):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(0.06, 0.96, title, fontsize=18, fontweight="bold", va="top", ha="left")

    y = 0.92
    for line in wrap_lines(lines):
        if y < 0.07:
            break
        fontsize = 12 if not line.startswith("    ") else 11
        ax.text(0.07, y, line, fontsize=fontsize, va="top", ha="left")
        y -= 0.026 if line else 0.018

    ax.text(
        0.06,
        0.03,
        f"Collision_sim 구조/가이드 함수 분석 | 페이지 {page_no}",
        fontsize=10,
        color="dimgray",
        ha="left",
        va="bottom",
    )
    pdf.savefig(fig)
    plt.close(fig)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg = parse_simple_yaml(CONFIG_PATH)
    setup_korean_font()

    today = datetime.now().strftime("%Y-%m-%d %H:%M")

    d = {
        "encoder_dim": cfg.get("encoder_dim", 256),
        "tf_layer": cfg.get("tf_layer", 3),
        "sampling": cfg.get("sampling", "ddpm"),
        "dynamics_guidance_enabled": cfg.get("dynamics_guidance_enabled", True),
        "dynamics_guidance_scale": cfg.get("dynamics_guidance_scale", 0.03),
        "dynamics_guidance_start_ratio": cfg.get("dynamics_guidance_start_ratio", 0.0),
        "dynamics_guidance_inner_steps": cfg.get("dynamics_guidance_inner_steps", 1),
        "dynamics_guidance_max_grad_norm": cfg.get("dynamics_guidance_max_grad_norm", 1.5),
        "dynamics_guidance_min_speed": cfg.get("dynamics_guidance_min_speed", 0.5),
        "dynamics_guidance_max_accel": cfg.get("dynamics_guidance_max_accel", 6.0),
        "dynamics_guidance_max_jerk": cfg.get("dynamics_guidance_max_jerk", 8.0),
        "dynamics_guidance_max_yaw_rate": cfg.get("dynamics_guidance_max_yaw_rate", 0.6),
        "dynamics_guidance_max_curvature": cfg.get("dynamics_guidance_max_curvature", 0.25),
        "dynamics_guidance_max_lateral_accel": cfg.get("dynamics_guidance_max_lateral_accel", 4.5),
        "dynamics_guidance_max_slip_ratio": cfg.get("dynamics_guidance_max_slip_ratio", 3.0),
        "dynamics_guidance_reverse_tolerance": cfg.get("dynamics_guidance_reverse_tolerance", 0.25),
        "yaw_loss_enabled": cfg.get("yaw_loss_enabled", True),
        "yaw_loss_weight": cfg.get("yaw_loss_weight", 0.1),
        "yaw_loss_min_speed": cfg.get("yaw_loss_min_speed", 0.5),
    }

    pages = []
    pages.append(
        (
            "Collision_sim 모델/가이드 함수 분석 (한글)",
            [
                f"문서 생성 시각: {today}",
                "분석 범위: 현재 코드 기준(mid.py, models/autoencoder.py, models/diffusion.py, "
                "utils/trajectron_hypers.py, process_data_mat.py, single_integrator.py)",
                "",
                "[핵심 결론]",
                "1) 모델은 Trajectron 인코더 + Diffusion(속도 예측) + SingleIntegrator(위치 적분) 구조입니다.",
                "2) Dynamics Guidance는 샘플링 단계에서만 동작하며, 경로를 물리적으로 더 타당한 쪽으로 "
                "gradient 업데이트합니다.",
                "3) 현재는 yaw 보조 loss가 추가되어 학습 단계에서도 yaw 정합을 직접 개선합니다.",
                "",
                "[코드 근거 함수]",
                "- 학습 루프: mid.py `MID.train()`",
                "- 잠재벡터 추출/샘플링/손실: models/autoencoder.py",
                "- 확산 손실, 역샘플링, guidance 목적함수: models/diffusion.py",
                "- 상태/타깃 정의: utils/trajectron_hypers.py",
                "- yaw 전처리: process_data_mat.py `build_yaw_series`",
            ],
        )
    )

    pages.append(
        (
            "1. 전체 모델 구조",
            [
                "[입력/상태 정의]",
                "- state (입력 컨텍스트): position(x,y), velocity(x,y), acceleration(x,y), heading(yaw,yaw_rate)",
                "- pred_state (학습 타깃): velocity(x,y)",
                "- prediction_horizon=20, history=12",
                "",
                "[파이프라인]",
                "    MAT/txt -> process_data_mat.py -> pkl(Environment/Scene/Node)",
                "    -> Trajectron(MGCVAE) get_latent(batch,node_type)=context f",
                "    -> DiffusionTraj가 표준화 속도 y_st를 생성",
                "    -> std 복원으로 물리 속도 v_phys",
                "    -> SingleIntegrator: p_t = p_0 + dt * cumsum(v_t)",
                "    -> (옵션) yaw/speed 계산 후 반환",
                "",
                "[디코더 네트워크]",
                f"- diffnet: TransformerConcatLinear, context_dim={d['encoder_dim']}, tf_layer={d['tf_layer']}",
                "- diffusion step 수: 100, variance schedule: linear(beta_T=5e-2)",
            ],
        )
    )

    pages.append(
        (
            "2. 학습 손실(현재 구조)",
            [
                "[기본 diffusion 손실]",
                "x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * eps",
                "eps_theta = net(x_t, t/T, context)",
                "L_diff = MSE(eps_theta, eps)",
                "",
                "[x0 복원]",
                "x0_hat = (x_t - sqrt(1-alpha_bar_t)*eps_theta) / sqrt(alpha_bar_t)",
                "",
                "[yaw 보조 손실(추가됨)]",
                "v_hat = x0_hat * std_v,   v_gt = y_st * std_v",
                "psi_hat = atan2(v_hat_y, v_hat_x), psi_gt = atan2(v_gt_y, v_gt_x)",
                "L_yaw = mean_M [ 1 - cos( wrap(psi_hat - psi_gt) ) ]",
                "M: (pred_speed > min_speed) OR (gt_speed > min_speed)",
                "",
                f"최종 손실: L_total = L_diff + lambda_yaw * L_yaw,   lambda_yaw={d['yaw_loss_weight']}",
                f"yaw_loss_enabled={d['yaw_loss_enabled']}, yaw_loss_min_speed={d['yaw_loss_min_speed']}",
                "",
                "[중요 포인트]",
                "- Dynamics Guidance는 학습 손실 안에 직접 들어가지 않고 샘플링 중 경로 보정으로 적용됩니다.",
            ],
        )
    )

    pages.append(
        (
            "3. Dynamics Guidance 수식과 의미",
            [
                "[샘플링 기본 업데이트(ddpm)]",
                "x_{t-1} = c0 * (x_t - c1 * eps_theta) + sigma_t * z",
                "",
                "[가이드 적용]",
                "x_{t-1} <- x_{t-1} - lambda_g * grad_x G(x_{t-1})",
                f"lambda_g={d['dynamics_guidance_scale']}, inner_steps={d['dynamics_guidance_inner_steps']}, "
                f"max_grad_norm={d['dynamics_guidance_max_grad_norm']}",
                "",
                "[목적함수 G의 구성(soft constraint)]",
                "- 가속도 위반: relu(||a||-a_max)^2",
                "- jerk 위반: relu(||j||-j_max)^2",
                "- yaw_rate 위반: relu(|dpsi/dt|-yaw_rate_max)^2",
                "- 곡률 위반: relu((|dpsi/dt|/v)-kappa_max)^2",
                "- 횡가속도 위반: relu((|dpsi/dt|*v)-a_lat_max)^2",
                "- slip 유사 항: relu((a_lat/(|a_long|+eps))-slip_max)^2",
                "- 역주행 유사 항: relu(-(v·dir + tol))^2",
                "",
                "[현재 임계값]",
                f"a_max={d['dynamics_guidance_max_accel']}, j_max={d['dynamics_guidance_max_jerk']}, "
                f"yaw_rate_max={d['dynamics_guidance_max_yaw_rate']}",
                f"kappa_max={d['dynamics_guidance_max_curvature']}, a_lat_max={d['dynamics_guidance_max_lateral_accel']}, "
                f"slip_max={d['dynamics_guidance_max_slip_ratio']}",
                f"reverse_tol={d['dynamics_guidance_reverse_tolerance']}, min_speed={d['dynamics_guidance_min_speed']}",
            ],
        )
    )

    pages.append(
        (
            "4. 속도-요 제약의 결합 해석",
            [
                "[질문: 속도 10과 100에서 yaw 제약이 같은가?]",
                "- yaw_rate 상한 자체(max_yaw_rate)는 상수라서 동일합니다.",
                "- 하지만 곡률(kappa = yaw_rate / v)과 횡가속도(a_lat = yaw_rate * v)는 속도 v에 직접 의존합니다.",
                "- 따라서 고속일수록 같은 yaw_rate라도 a_lat 제약을 더 빨리 위반할 수 있습니다.",
                "",
                "[실무 해석]",
                "- 현재 구조는 'yaw_rate 단독 제약 + 속도 연동 제약(곡률/횡가속도)'의 조합입니다.",
                "- 즉, 완전히 속도 무관하지도 않고 완전히 속도 종속도 아닌 하이브리드 제약입니다.",
                "",
                "[적용 타이밍]",
                f"- guidance_enabled={d['dynamics_guidance_enabled']}, start_ratio={d['dynamics_guidance_start_ratio']}",
                "- start_ratio=0.0이면 사실상 전체 reverse step에서 guidance가 적용됩니다.",
            ],
        )
    )

    pages.append(
        (
            "5. 출력 텐서와 한계/개선 포인트",
            [
                "[generate(return_dynamics=True) 출력]",
                "- position: [S, B, T, 2]",
                "- velocity: [S, B, T, 2]",
                "- speed: [S, B, T]",
                "- yaw: [S, B, T]  (velocity에서 atan2로 계산, 저속 구간은 이전 yaw 유지)",
                "",
                "[현재 구조의 장점]",
                "- 기존 MID 코드와 호환되며, 충돌 시나리오에 필요한 동역학 제약을 샘플링에 쉽게 주입 가능",
                "- yaw 보조 loss로 학습 단계에서도 방향성 정합 개선",
                "",
                "[현재 구조의 한계]",
                "- guidance는 hard constraint가 아니라 soft penalty이므로 위반이 0으로 보장되지는 않음",
                "- 동역학 모델이 SingleIntegrator라 차량 조향계/타이어 모델을 직접 반영하지 않음",
                "- slip 항은 물리 기반 타이어 모델이 아닌 근사 지표",
                "",
                "[다음 개선 우선순위 제안]",
                "1) 고속 주행 비율이 크면 max_yaw_rate보다 a_lat/kappa 항 가중치 조정 우선",
                "2) 차량 거동을 더 실제처럼 만들려면 Unicycle/Bicycle 동역학으로 교체 검토",
                "3) 충돌 합성 품질을 위해 TTC/상대속도 기반 충돌 score를 guidance 항으로 추가",
            ],
        )
    )

    with PdfPages(OUT_PATH) as pdf:
        for i, (title, lines) in enumerate(pages, start=1):
            draw_page(pdf, title, lines, i)

    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()

