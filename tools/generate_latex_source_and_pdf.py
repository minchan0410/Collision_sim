from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output"
TEX_PATH = OUT_DIR / "collision_sim_formulation_ko.tex"
PDF_PATH = OUT_DIR / "collision_sim_formulation_ko_latex_style.pdf"


def setup_korean_font():
    candidates = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
    ]
    for p in candidates:
        if p.exists():
            font_manager.fontManager.addfont(str(p))
            rcParams["font.family"] = font_manager.FontProperties(fname=str(p)).get_name()
            break
    rcParams["axes.unicode_minus"] = False


def write_tex_source():
    tex = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,bm}
\usepackage{kotex}
\usepackage{hyperref}
\title{Collision\_sim 수식 정리 (Diffusion + Dynamics Guidance + Bicycle Rollout)}
\author{}
\date{}
\begin{document}
\maketitle

\section*{1. Diffusion 학습}
\[
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I)
\]
\[
\mathcal{L}_{\mathrm{diff}}=\left\|\epsilon_\theta(x_t,t,c)-\epsilon\right\|_2^2
\]
\[
\hat{x}_0=\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t,c)}{\sqrt{\bar{\alpha}_t}}
\]

\section*{2. Yaw 보조 Loss}
\[
\hat{v}=\hat{x}_0\odot\sigma_v,\qquad v^{gt}=y_{st}\odot\sigma_v
\]
\[
\hat{d}=\frac{\hat{v}}{\max(\|\hat{v}\|,v_{\min}+\varepsilon)},\quad
d^{gt}=\frac{v^{gt}}{\max(\|v^{gt}\|,v_{\min}+\varepsilon)}
\]
\[
\mathcal{L}_{\mathrm{yaw}}
=\frac{1}{|M|}\sum_{(b,\tau)\in M}\left(1-\hat{d}_{b,\tau}^{\top}d_{b,\tau}^{gt}\right),
\quad
M=\{(b,\tau)\mid \|\hat{v}_{b,\tau}\|>v_{\min}\land \|v^{gt}_{b,\tau}\|>v_{\min}\}
\]
\[
\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{diff}}+\lambda_{\mathrm{yaw}}\mathcal{L}_{\mathrm{yaw}}
\]

\section*{3. Dynamics Guidance (샘플링 중)}
\[
x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t,c)\right)+\sigma_t z
\]
\[
x_{t-1}\leftarrow x_{t-1}-\lambda_g\nabla_x G(x_{t-1})
\]
\[
G=\sum_i w_i P_i
\]
\[
P_{\mathrm{acc}}=\mathrm{mean}\left[\max(0,\|a\|-a_{\max})^2\right],\quad
P_{\mathrm{jerk}}=\mathrm{mean}\left[\max(0,\|j\|-j_{\max})^2\right]
\]
\[
\dot{\psi}=\frac{|\mathrm{wrap}(\psi_{k+1}-\psi_k)|}{\Delta t},\quad
\kappa=\frac{\dot{\psi}}{\bar{v}}
\]
\[
P_{\mathrm{yaw}}=\mathrm{mean}\left[\max(0,\dot{\psi}-\dot{\psi}_{\max}^{\mathrm{lim}})^2\right],\quad
P_{\kappa}=\mathrm{mean}\left[\max(0,\kappa-\kappa_{\max})^2\right]
\]

\section*{4. Bicycle 기반 제한}
\[
\kappa_{\mathrm{bike}}=\frac{\tan(\delta_{\max})}{L}
\]
\[
\dot{\psi}_{\max}^{\mathrm{lim}}(v)=\min\left(\dot{\psi}_{\max},v\kappa_{\mathrm{bike}}\right)
\]
\[
\Delta\psi\leftarrow\mathrm{clip}\left(\Delta\psi,-\dot{\psi}_{\max}^{\mathrm{lim}}\Delta t,\dot{\psi}_{\max}^{\mathrm{lim}}\Delta t\right)
\]
\[
v_x=s\cos\psi,\quad v_y=s\sin\psi,\quad p_{k+1}=p_k+v_k\Delta t
\]

\end{document}
""".lstrip()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEX_PATH.write_text(tex, encoding="utf-8")


def add_page(pdf, title, body_lines, equations):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.06, 0.96, title, fontsize=18, fontweight="bold", va="top")

    y = 0.90
    for line in body_lines:
        ax.text(0.08, y, line, fontsize=11)
        y -= 0.038

    y -= 0.01
    for eq in equations:
        ax.text(0.10, y, eq, fontsize=14)
        y -= 0.085
        if y < 0.1:
            break

    pdf.savefig(fig)
    plt.close(fig)


def render_pdf():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with PdfPages(PDF_PATH) as pdf:
        add_page(
            pdf,
            "Collision_sim 수식 정리 (LaTeX 스타일)",
            [
                "이 PDF는 수식을 LaTeX 문법으로 정리한 버전입니다.",
                "동시에 동일 내용을 정식 .tex 소스로도 생성했습니다.",
            ],
            [
                r"$x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,\ \epsilon\sim\mathcal{N}(0,I)$",
                r"$\mathcal{L}_{diff}=\left\|\epsilon_\theta(x_t,t,c)-\epsilon\right\|_2^2$",
                r"$\hat{x}_0=\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t,c)}{\sqrt{\bar{\alpha}_t}}$",
            ],
        )

        add_page(
            pdf,
            "1) Yaw 보조 Loss",
            [
                "학습시 yaw 정합을 직접 반영하는 보조 손실:",
            ],
            [
                r"$\hat{v}=\hat{x}_0\odot\sigma_v,\ \ v^{gt}=y_{st}\odot\sigma_v$",
                r"$\hat{d}=\frac{\hat{v}}{\max(\|\hat{v}\|,v_{min}+\varepsilon)},\ d^{gt}=\frac{v^{gt}}{\max(\|v^{gt}\|,v_{min}+\varepsilon)}$",
                r"$\mathcal{L}_{yaw}=\frac{1}{|M|}\sum_{(b,\tau)\in M}\left(1-\hat{d}_{b,\tau}^{\top}d_{b,\tau}^{gt}\right)$",
                r"$\mathcal{L}_{total}=\mathcal{L}_{diff}+\lambda_{yaw}\mathcal{L}_{yaw}$",
            ],
        )

        add_page(
            pdf,
            "2) Dynamics Guidance",
            [
                "샘플링 중간 스텝에서 gradient 보정:",
            ],
            [
                r"$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t,c)\right)+\sigma_t z$",
                r"$x_{t-1}\leftarrow x_{t-1}-\lambda_g\nabla_x G(x_{t-1})$",
                r"$G=\sum_i w_iP_i,\ \ P_i=\mathrm{ReLU}(\mathrm{violation}_i)^2$",
                r"$P_{yaw}=\mathrm{mean}\left[\max(0,\dot{\psi}-\dot{\psi}_{max}^{lim})^2\right]$",
            ],
        )

        add_page(
            pdf,
            "3) Bicycle 기반 제한 + Rollout",
            [
                "회전반경/조향각 기반 제한과 최종 궤적 보정:",
            ],
            [
                r"$\kappa_{bike}=\frac{\tan(\delta_{max})}{L}$",
                r"$\dot{\psi}_{max}^{lim}(v)=\min\left(\dot{\psi}_{max},v\kappa_{bike}\right)$",
                r"$\Delta\psi\leftarrow \mathrm{clip}\left(\Delta\psi,-\dot{\psi}_{max}^{lim}\Delta t,+\dot{\psi}_{max}^{lim}\Delta t\right)$",
                r"$v_x=s\cos\psi,\ \ v_y=s\sin\psi,\ \ p_{k+1}=p_k+v_k\Delta t$",
            ],
        )


def main():
    setup_korean_font()
    write_tex_source()
    render_pdf()
    print(f"saved tex: {TEX_PATH}")
    print(f"saved pdf: {PDF_PATH}")


if __name__ == "__main__":
    main()

