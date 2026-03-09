"""
论文实验脚本 - 可旋转天线增强网络 ISAC 系统
生成三张论文图（含三条基线对比）：
  Fig1: Rate-CRB(α) Pareto 权衡曲线（全方案汇总）
  Fig2: 旋转天线 vs 固定天线(φ=0) vs 随机旋转角
  Fig3: 多基站(U=3) vs 单基站(U=1)

运行方式（与 isac_rotatable_antenna.py 同目录）：
  python run_experiments.py
"""

import sys, os, io, warnings, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

warnings.filterwarnings('ignore')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ── 导入主模块函数（不执行 __main__）──
_src = open(os.path.join(SCRIPT_DIR, 'isac_rotatable_antenna.py')).read()
exec(_src.split("if __name__")[0], globals())

print("=" * 60)
print("论文实验脚本：可旋转天线 ISAC 系统")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
#  公共系统参数
# ══════════════════════════════════════════════════════════════
U, K, M, N = 3, 2, 4, 4
bs_pos   = np.array([[0., 0.], [50., 0.], [25., 43.3]])
user_pos = np.array([[20., 30.], [40., 10.]])
alpha_true, r_true = np.deg2rad(30), 100.0
beta     = np.array([1.0 + 0.5j, 0.8 - 0.3j, 0.9 + 0.2j])
P_max    = np.array([1.0, 1.0, 1.0])
sigma2   = 1e-2
sigma2_k = np.array([1e-2, 1e-2])
phi_min, phi_max_v = -np.pi / 4, np.pi / 4

d0 = 30.0
tau = np.zeros((U, K))
for u in range(U):
    for k in range(K):
        d_uk = np.sqrt((bs_pos[u, 0] - user_pos[k, 0]) ** 2 +
                       (bs_pos[u, 1] - user_pos[k, 1]) ** 2)
        # 使用 max(d_uk, d0) 确保 tau <= 1（路径增益不超过参考距离处）
        tau[u][k] = (d0 / max(d_uk, d0)) ** 2

EPS_R_FIXED = 50.0  # 距离 CRB 阈值固定，扫描角度 CRB

# AO 超参数
AO_PARAMS = dict(
    T_max=35, delta_AO=1e-2, eps_abs=1e-3, eps_AO=1e-4,
    rho_init=0.0, method='PSO',
    pso_params={'P_pso': 15, 'T_pso': 40}
)

BASE = dict(
    bs_pos=bs_pos, user_pos=user_pos, tau=tau,
    P_max=P_max, beta=beta, sigma2=sigma2, sigma2_k=sigma2_k,
    alpha_true=alpha_true, r_true=r_true,
    M=M, N=N, U=U, K=K,
    phi_min=phi_min, phi_max=phi_max_v,
    lambda_alpha=10.0, lambda_r=10.0,
    **AO_PARAMS
)

# 绘图样式
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 12,
    'axes.labelsize': 13, 'axes.titlesize': 13,
    'legend.fontsize': 11, 'axes.grid': True,
    'grid.alpha': 0.35, 'lines.linewidth': 2.0,
    'lines.markersize': 7,
})
COLORS  = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
MARKERS = ['o', 's', '^', 'D']

# 扫描的 eps_alpha 序列（14个对数均匀点）
# 下限 1e-5 rad² 对应紧约束（高精度感知），上限 10 rad² 对应松约束（低精度感知）
# 原 1e-7 对该系统几乎不可达（导致大量跳过），调整为合理范围
EPS_ALPHA_SWEEP = np.logspace(-5, 1, 14)


# ══════════════════════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════════════════════
def run_silent(params):
    """静默运行 algo11_ao，返回 (R_star, CRB_alpha) 或 (None, None)"""
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            res = algo11_ao(**params)
        R_v, C_v = res['R_star'], res['CRB_alpha']
        if np.isfinite(R_v) and np.isfinite(C_v) and R_v > 0:
            return float(R_v), float(C_v)
    except Exception:
        pass
    return None, None


def sweep(base_params, label):
    """扫描 eps_alpha（从紧到松，每次独立冷启动），返回 (R_list, CRB_list)"""
    Rs, Cs = [], []
    n = len(EPS_ALPHA_SWEEP)
    for i, eps_a in enumerate(EPS_ALPHA_SWEEP):
        t0 = time.time()
        p = {**base_params, 'eps_alpha': float(eps_a), 'eps_r': EPS_R_FIXED}
        R_v, C_v = run_silent(p)
        elapsed = time.time() - t0
        if R_v is not None:
            Rs.append(R_v); Cs.append(C_v)
            print(f"  [{label}] {i+1}/{n}  eps={eps_a:.1e}  "
                  f"R={R_v:.3f}  CRB={C_v:.2e}  ({elapsed:.0f}s)")
        else:
            print(f"  [{label}] {i+1}/{n}  eps={eps_a:.1e}  skip  ({elapsed:.0f}s)")
    return Rs, Cs


def sorted_plot(ax, Cs, Rs, color, marker, linestyle, label):
    """绘制帕累托边界：按 CRB 排序后取累积最大值，保证 R 单调不减"""
    if not Cs:
        return
    idx = np.argsort(Cs)
    Cs_sorted = np.array(Cs)[idx]
    Rs_sorted = np.array(Rs)[idx]
    # 帕累托包络：若松约束点陷入差的局部最优，用紧约束处更好结果替代
    Rs_pareto = np.maximum.accumulate(Rs_sorted)
    ax.plot(Cs_sorted, Rs_pareto,
            color=color, marker=marker, linestyle=linestyle, label=label)


# ══════════════════════════════════════════════════════════════
#  方案 A: Proposed — 多基站 + 可旋转天线（联合优化）
# ══════════════════════════════════════════════════════════════
print("\n[1/4] Proposed: Multi-BS + Optimized Rotation")
R_prop, C_prop = sweep(BASE, "Proposed")

# ══════════════════════════════════════════════════════════════
#  方案 B: 固定天线 φ = 0（不优化旋转角）
# ══════════════════════════════════════════════════════════════
print("\n[2/4] Baseline B: Fixed Antenna (φ = 0)")
BASE_FIXED = {**BASE, 'phi_min': 0.0, 'phi_max': 0.0}
R_fixed, C_fixed = sweep(BASE_FIXED, "Fixed")

# ══════════════════════════════════════════════════════════════
#  方案 C: 随机旋转角（φ 固定在非最优值，不优化）
#   取三基站角度：φ = [20°, -15°, 10°]
#   通过设 phi_min = phi_max = 单一值（均值）实现锁定
# ══════════════════════════════════════════════════════════════
print("\n[3/4] Baseline C: Random Rotation (φ = 20°, -15°, 10°, not optimized)")
# 均值 ≈ 5°，PSO 只能选唯一点
phi_rand_mean = float(np.mean(np.deg2rad([20., -15., 10.])))
BASE_RAND = {**BASE,
             'phi_min': phi_rand_mean - 1e-8,
             'phi_max': phi_rand_mean + 1e-8}
R_rand, C_rand = sweep(BASE_RAND, "Random-phi")

# ══════════════════════════════════════════════════════════════
#  方案 D: 单基站 U = 1（等效总功率 = 3W）
# ══════════════════════════════════════════════════════════════
print("\n[4/4] Baseline D: Single-BS (U=1, P=3W)")
BASE_U1 = dict(
    bs_pos=bs_pos[:1, :],
    user_pos=user_pos,
    tau=tau[:1, :],
    P_max=np.array([3.0]),       # 总功率与多基站相同
    beta=beta[:1],
    sigma2=sigma2, sigma2_k=sigma2_k,
    alpha_true=alpha_true, r_true=r_true,
    M=M, N=N, U=1, K=K,
    phi_min=phi_min, phi_max=phi_max_v,
    lambda_alpha=10.0, lambda_r=10.0,
    **AO_PARAMS
)
R_u1, C_u1 = sweep(BASE_U1, "Single-BS")

# ── 保存原始数据 ──
np.savez(os.path.join(SCRIPT_DIR, 'exp_results.npz'),
         R_prop=R_prop, C_prop=C_prop,
         R_fixed=R_fixed, C_fixed=C_fixed,
         R_rand=R_rand, C_rand=C_rand,
         R_u1=R_u1, C_u1=C_u1)
print("\n原始数据已保存至 exp_results.npz")


# ══════════════════════════════════════════════════════════════
#  Fig 1: Rate-CRB(α) Pareto 权衡曲线（全方案）
# ══════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(7.5, 5.5))

sorted_plot(ax1, C_prop,  R_prop,  COLORS[0], MARKERS[0], '-',  'Proposed (Multi-BS + Opt. Rotation)')
sorted_plot(ax1, C_fixed, R_fixed, COLORS[1], MARKERS[1], '--', 'Fixed Antenna (φ=0)')
sorted_plot(ax1, C_rand,  R_rand,  COLORS[2], MARKERS[2], '-.',  'Random Rotation')
sorted_plot(ax1, C_u1,    R_u1,    COLORS[3], MARKERS[3], ':',  'Single-BS (U=1)')

ax1.set_xlabel('CRB(α) [rad²]')
ax1.set_ylabel('Sum Rate R* [bps/Hz]')
ax1.set_title('Rate–CRB(α) Tradeoff (Pareto Boundary)')
ax1.set_xscale('log')
ax1.legend(loc='upper left', framealpha=0.85)
fig1.tight_layout()
path1 = os.path.join(SCRIPT_DIR, 'fig1_rate_crb_tradeoff.png')
fig1.savefig(path1, dpi=150, bbox_inches='tight')
print(f"→ 保存 {path1}")


# ══════════════════════════════════════════════════════════════
#  Fig 2: 旋转天线 vs 固定天线 vs 随机旋转
# ══════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(7.5, 5.5))

sorted_plot(ax2, C_prop,  R_prop,  COLORS[0], MARKERS[0], '-',   'Proposed (Optimized Rotation)')
sorted_plot(ax2, C_fixed, R_fixed, COLORS[1], MARKERS[1], '--',  'Fixed Antenna (φ=0)')
sorted_plot(ax2, C_rand,  R_rand,  COLORS[2], MARKERS[2], '-.',   'Random Rotation (φ=const)')

ax2.set_xlabel('CRB(α) [rad²]')
ax2.set_ylabel('Sum Rate R* [bps/Hz]')
ax2.set_title('Rotatable vs Fixed vs Random Antenna Array')
ax2.set_xscale('log')
ax2.legend(framealpha=0.85)
fig2.tight_layout()
path2 = os.path.join(SCRIPT_DIR, 'fig2_rotatable_vs_fixed.png')
fig2.savefig(path2, dpi=150, bbox_inches='tight')
print(f"→ 保存 {path2}")


# ══════════════════════════════════════════════════════════════
#  Fig 3: 多基站 vs 单基站
# ══════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(7.5, 5.5))

sorted_plot(ax3, C_prop, R_prop, COLORS[0], MARKERS[0], '-',
            f'Multi-BS (U={U}, P_total={int(P_max.sum())}W)')
sorted_plot(ax3, C_u1,   R_u1,   COLORS[3], MARKERS[3], '--',
            f'Single-BS (U=1, P_total={int(BASE_U1["P_max"].sum())}W)')

ax3.set_xlabel('CRB(α) [rad²]')
ax3.set_ylabel('Sum Rate R* [bps/Hz]')
ax3.set_title('Multi-BS vs Single-BS Cooperation Gain')
ax3.set_xscale('log')
ax3.legend(framealpha=0.85)
fig3.tight_layout()
path3 = os.path.join(SCRIPT_DIR, 'fig3_multiBs_vs_singleBs.png')
fig3.savefig(path3, dpi=150, bbox_inches='tight')
print(f"→ 保存 {path3}")


print("\n" + "=" * 60)
print("✓ 全部实验完成！生成文件：")
for f in ['fig1_rate_crb_tradeoff.png',
          'fig2_rotatable_vs_fixed.png',
          'fig3_multiBs_vs_singleBs.png',
          'exp_results.npz']:
    print(f"  {f}")
print("=" * 60)
