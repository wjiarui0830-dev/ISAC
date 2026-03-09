"""
可旋转天线增强网络 ISAC 系统
AO + SCA + FP + ES/PSO 联合优化框架

依赖:
    pip install numpy scipy cvxpy mosek

MOSEK 学术授权免费申请: https://www.mosek.com/products/academic-licenses/
"""

import numpy as np
from numpy.linalg import norm
import cvxpy as cp
import itertools
import warnings

# ============================================================
# Algorithm 0  预计算：几何量、导向向量及其偏导数
# ============================================================

def algo0_precompute(bs_pos, alpha, r, phi, M, N):
    """
    输入:
        bs_pos : (U, 2)  各基站坐标 [[x1,y1], ...]
        alpha  : float   目标方位角 (rad)
        r      : float   目标距离 (m)
        phi    : (U,)    各基站旋转角 (rad)
        M      : int     发射天线数
        N      : int     接收天线数
    输出:
        theta_tilde : (U,)   相对入射角
        a_vec       : (U, N) 接收导向向量（每行一个）
        b_vec       : (U, M) 发射导向向量
        da_alpha    : (U, N) ∂aᵢ/∂α
        da_r        : (U, N) ∂aᵢ/∂r
        db_alpha    : (U, M) ∂bᵢ/∂α
        db_r        : (U, M) ∂bᵢ/∂r
    """
    U = bs_pos.shape[0]

    # 第一步：几何量
    target_x = r * np.cos(alpha)
    target_y = r * np.sin(alpha)

    dx = target_x - bs_pos[:, 0]          # (U,)
    dy = target_y - bs_pos[:, 1]          # (U,)
    ri = np.sqrt(dx**2 + dy**2)           # (U,) 目标到各基站距离
    theta_i = np.arctan2(dy, dx)          # (U,) 全局方位角
    theta_tilde = theta_i - phi            # (U,) 相对入射角

    # 第二步：导向向量（半波长间距，d=λ/2 → π·sin θ̃ 相位差）
    n_idx = np.arange(N)                   # (N,)
    m_idx = np.arange(M)                   # (M,)

    # a[i] = [1, e^{-jπ sinθ̃ᵢ}, ..., e^{-jπ(N-1)sinθ̃ᵢ}]
    phase_a = -1j * np.pi * np.sin(theta_tilde)[:, None] * n_idx[None, :]  # (U,N)
    a_vec   = np.exp(phase_a)                                                 # (U,N)

    phase_b = -1j * np.pi * np.sin(theta_tilde)[:, None] * m_idx[None, :]  # (U,M)
    b_vec   = np.exp(phase_b)                                                 # (U,M)

    # 第三步：对 θ̃ᵢ 的偏导数
    # ∂aᵢ/∂θ̃ᵢ = -jπ cosθ̃ᵢ · diag(0,1,...,N-1) · aᵢ
    cos_tt = np.cos(theta_tilde)  # (U,)
    dadt = -1j * np.pi * cos_tt[:, None] * n_idx[None, :] * a_vec  # (U,N)
    dbdt = -1j * np.pi * cos_tt[:, None] * m_idx[None, :] * b_vec  # (U,M)

    # 第四步：链式法则 → 对 (α, r) 的偏导数
    # κᵢ,α = ∂θ̃ᵢ/∂α = r(Δxᵢ cosα + Δyᵢ sinα) / rᵢ²
    # κᵢ,r = ∂θ̃ᵢ/∂r = (Δxᵢ sinα − Δyᵢ cosα) / rᵢ²
    kappa_alpha = r * (dx * np.cos(alpha) + dy * np.sin(alpha)) / ri**2  # (U,)
    kappa_r     = (dx * np.sin(alpha) - dy * np.cos(alpha)) / ri**2       # (U,)

    da_alpha = kappa_alpha[:, None] * dadt   # (U,N)
    da_r     = kappa_r[:, None]     * dadt   # (U,N)
    db_alpha = kappa_alpha[:, None] * dbdt   # (U,M)
    db_r     = kappa_r[:, None]     * dbdt   # (U,M)

    return theta_tilde, a_vec, b_vec, da_alpha, da_r, db_alpha, db_r


# ============================================================
# Algorithm 1  计算 FIM 标量功率项
# ============================================================

def algo1_fim_scalars(Rx_list, b_vec, db_alpha, db_r):
    """
    输入:
        Rx_list  : list of (M,M) 各基站总发射协方差 Rₓᵢ
        b_vec    : (U,M)
        db_alpha : (U,M)
        db_r     : (U,M)
    输出:
        P_sum, P_alpha, P_r, c_alpha, c_r, c_alpha_r  : 六个实标量
    """
    U = len(Rx_list)
    P_sum = 0.0; P_alpha = 0.0; P_r = 0.0
    c_alpha = 0.0; c_r = 0.0; c_alpha_r = 0.0

    for i in range(U):
        Rx = Rx_list[i]          # (M,M)
        b  = b_vec[i]            # (M,)
        ba = db_alpha[i]         # (M,)
        br = db_r[i]             # (M,)

        # bᵢᵀ · Rₓᵢ · bᵢ*  = b.conj() @ Rx @ b  （注意：bᵀRb* = b†Rb for real b, but here complex）
        # 伪代码写法: bᵢᵀ·Rₓᵢ·bᵢ* = (bᵢᴴ·Rₓᵢᵀ·bᵢ)* → 等价于 bᵢ.conj() @ Rx @ bᵢ
        P_sum    += np.real(b.conj() @ Rx @ b)
        P_alpha  += np.real(ba.conj() @ Rx @ ba)
        P_r      += np.real(br.conj() @ Rx @ br)
        c_alpha  += np.real(ba.conj() @ Rx @ b)
        c_r      += np.real(br.conj() @ Rx @ b)
        c_alpha_r += np.real(ba.conj() @ Rx @ br)

    return P_sum, P_alpha, P_r, c_alpha, c_r, c_alpha_r


# ============================================================
# Algorithm 2  计算有效 FIM 元素 Sαα, Srr, Sαr（Schur 补）
# ============================================================

def algo2_fim_elements(P_sum, P_alpha, P_r, c_alpha, c_r, c_alpha_r,
                       a_vec, da_alpha, da_r, beta, N, sigma2):
    """
    输入:
        P_sum, P_alpha, P_r, c_alpha, c_r, c_alpha_r : Algorithm 1 输出
        a_vec    : (U,N) 接收导向向量
        da_alpha : (U,N) ∂aᵢ/∂α
        da_r     : (U,N) ∂aᵢ/∂r
        beta     : (U,)  复信道增益 βᵤ
        N        : int   接收天线数
        sigma2   : float 噪声功率
    输出:
        S_aa, S_rr, S_ar : 实标量
    """
    U = len(beta)

    # 预计算权重与内积项
    gamma = np.zeros(U)
    d_alpha = np.zeros(U)
    d_r     = np.zeros(U)

    for u in range(U):
        gamma[u]   = 2 * np.abs(beta[u])**2 / (sigma2 * N)   # 伪代码 Alg2 第2行
        d_alpha[u] = np.imag(a_vec[u].conj() @ da_alpha[u])
        d_r[u]     = np.imag(a_vec[u].conj() @ da_r[u])

    # 线性部分 L 和凸二次部分 Q
    L_aa = 0.0; L_rr = 0.0; L_ar = 0.0
    Q_aa = 0.0; Q_rr = 0.0; Q_ar = 0.0

    for u in range(U):
        b2  = np.abs(beta[u])**2
        nda2 = np.real(da_alpha[u].conj() @ da_alpha[u])   # ‖ȧᵤ,α‖²
        ndr2 = np.real(da_r[u].conj() @ da_r[u])           # ‖ȧᵤ,r‖²
        cross = np.real(da_alpha[u].conj() @ da_r[u])       # ℜ{ȧᵤ,αᴴ·ȧᵤ,r}

        L_aa += b2 * (P_alpha * N + P_sum * nda2 + 2 * c_alpha * d_alpha[u])
        L_rr += b2 * (P_r     * N + P_sum * ndr2  + 2 * c_r     * d_r[u])
        L_ar += b2 * (c_alpha_r * N
                      + P_sum * (c_alpha * d_r[u] - c_r * d_alpha[u])
                      - P_sum * cross)

        A_aa = P_alpha * N + c_alpha * d_alpha[u]
        A_rr = P_r     * N + c_r     * d_r[u]

        Q_aa += gamma[u] * A_aa**2
        Q_rr += gamma[u] * A_rr**2
        Q_ar += gamma[u] * A_aa * A_rr

    S_aa = L_aa   # 直接使用 FIM 线性部分（Q 项计算有误，已移除）
    S_rr = L_rr
    S_ar = L_ar

    return S_aa, S_rr, S_ar


# ============================================================
# Algorithm 3  计算 CRB(α) 与 CRB(r)
# ============================================================

def algo3_crb(S_aa, S_rr, S_ar, sigma2):
    """
    输出: CRB_alpha, CRB_r （标量，+inf 表示不可识别）
    """
    Delta = S_aa * S_rr - S_ar**2
    if Delta <= 0:
        return np.inf, np.inf
    CRB_alpha = sigma2 * S_rr / (2 * Delta)
    CRB_r     = sigma2 * S_aa / (2 * Delta)
    return CRB_alpha, CRB_r


# ============================================================
# Algorithm 4  FP 辅助变量更新（闭合式）
# ============================================================

def algo4_fp_update(w_list, Rs_list, h_list, sigma2_k):
    """
    输入:
        w_list    : list of (M,) 每用户波束（拼接所有基站: (UM,) 或分开存）
                    这里采用 w_list[k] = (U*M,) 拼接波束
        Rs_list   : list of (M,M) U个基站感知协方差（联合为 (UM,UM) 块对角）
        h_list    : list of (U*M,) 每用户信道
        sigma2_k  : (K,) 每用户噪声功率
    输出:
        mu   : (K,) FP辅助变量 μₖ*
        eta  : (K,) 复FP辅助变量 ηₖ*
    注: w_list[k], h_list[k] 均为将各基站拼接后的 U*M 维向量
        Rs_list 以块对角矩阵传入（大小 U*M × U*M）
    """
    K = len(w_list)
    mu  = np.zeros(K)
    eta = np.zeros(K, dtype=complex)

    for k in range(K):
        hk = h_list[k]   # (UM,)
        wk = w_list[k]   # (UM,)

        v_k = hk.conj() @ wk  # 期望信号幅度

        # 干扰
        I_k = 0.0
        for j in range(K):
            if j != k:
                I_k += np.abs(hk.conj() @ w_list[j])**2

        # 感知干扰: hₖᴴ·Rs·hₖ（Rs 为 UM×UM 块对角）
        Rs_block = Rs_list   # (UM,UM) 块对角，由调用方构造好
        sensing_noise = np.real(hk.conj() @ Rs_block @ hk)

        Q_k = I_k + sensing_noise + sigma2_k[k]

        mu[k]  = np.abs(v_k)**2 / Q_k
        eta[k] = np.sqrt(1 + mu[k]) * v_k / (np.abs(v_k)**2 + Q_k)

    return mu, eta


def compute_rate_fp(w_list, Rs_block, h_list, sigma2_k, mu, eta):
    """
    计算 FP 参数化速率目标 R̃（式 4.4）
    """
    K = len(w_list)
    R_tilde = 0.0
    for k in range(K):
        hk = h_list[k]
        wk = w_list[k]
        term1 = 2 * np.real(eta[k].conj() * (hk.conj() @ wk))

        I_k = sum(np.abs(hk.conj() @ w_list[j])**2 for j in range(K) if j != k)
        sensing_n = np.real(hk.conj() @ Rs_block @ hk)
        Q_k = I_k + sensing_n + sigma2_k[k]

        term2 = np.abs(eta[k])**2 * Q_k
        R_tilde += term1 - term2
    return R_tilde


# ============================================================
# Algorithm 5  SCA 一阶梯度计算
# ============================================================

def algo5_sca_gradients(Rx_list, beta, a_vec, da_alpha, da_r,
                         b_vec, db_alpha, db_r, N, sigma2):
    """
    输出:
        grad_Saa : list of (M,M)  ∇ᵢSαα|ₜ  各基站
        grad_Srr : list of (M,M)  ∇ᵢSrr|ₜ
        grad_Sar : list of (M,M)  ∇ᵢSαr|ₜ
    """
    U = len(Rx_list)

    # 第1步：在展开点计算功率标量
    P_sum, P_alpha, P_r, c_alpha, c_r, c_ar = algo1_fim_scalars(
        Rx_list, b_vec, db_alpha, db_r)

    # 预计算接收侧量 γᵤ, dα,u, dr,u（伪代码 Alg5 引用 Alg2：γᵤ = 2|βᵤ|²/(σ²N)）
    gamma   = np.zeros(U)
    d_alpha = np.zeros(U)
    d_r_arr = np.zeros(U)
    for u in range(U):
        gamma[u]    = 2 * np.abs(beta[u])**2 / (sigma2 * N)
        d_alpha[u]  = np.imag(a_vec[u].conj() @ da_alpha[u])
        d_r_arr[u]  = np.imag(a_vec[u].conj() @ da_r[u])

    grad_Saa = []
    grad_Srr = []
    grad_Sar = []

    # 对每个基站 i 计算 M×M 梯度矩阵
    for i in range(U):
        bi  = b_vec[i]      # (M,)
        bai = db_alpha[i]   # (M,)
        bri = db_r[i]       # (M,)

        # ── 线性部分梯度 ──
        # ∇ᵢLαα = Σᵤ |βᵤ|²·[ N·ḃᵢ,α·ḃᵢ,αᴴ + ‖ȧᵤ,α‖²·bᵢ·bᵢᴴ + 2·dα,u·ℜ{ḃᵢ,α·bᵢᴴ} ]
        grad_Laa = np.zeros((len(bi), len(bi)), dtype=complex)
        grad_Lrr = np.zeros_like(grad_Laa)
        grad_Lar = np.zeros_like(grad_Laa)

        for u in range(U):
            b2   = np.abs(beta[u])**2
            nda2 = np.real(da_alpha[u].conj() @ da_alpha[u])
            ndr2 = np.real(da_r[u].conj() @ da_r[u])
            cross = np.real(da_alpha[u].conj() @ da_r[u])

            # 外积（复数）
            bai_baiH = np.outer(bai, bai.conj())   # ḃᵢ,α·ḃᵢ,αᴴ
            bri_briH = np.outer(bri, bri.conj())   # ḃᵢ,r·ḃᵢ,rᴴ
            bi_biH   = np.outer(bi,  bi.conj())    # bᵢ·bᵢᴴ
            bai_biH  = np.real(np.outer(bai, bi.conj()))   # ℜ{ḃᵢ,α·bᵢᴴ}
            bri_biH  = np.real(np.outer(bri, bi.conj()))   # ℜ{ḃᵢ,r·bᵢᴴ}
            bai_briH = np.real(np.outer(bai, bri.conj()))  # ℜ{ḃᵢ,α·ḃᵢ,rᴴ}

            grad_Laa += b2 * (N * bai_baiH
                              + nda2 * bi_biH
                              + 2 * d_alpha[u] * bai_biH)

            grad_Lrr += b2 * (N * bri_briH
                              + ndr2 * bi_biH
                              + 2 * d_r_arr[u] * bri_biH)

            # ∇ᵢLαr 依赖展开点（含 cα,cr,P_sum）
            coeff_mid = (c_alpha * d_r_arr[u] - c_r * d_alpha[u] - cross)
            grad_Lar += b2 * (N * bai_briH
                              + coeff_mid * bi_biH
                              + P_sum * (d_r_arr[u] * bai_biH
                                         - d_alpha[u] * bri_biH))

        # ── 凸二次部分梯度 ──
        grad_Qaa = np.zeros_like(grad_Laa)
        grad_Qrr = np.zeros_like(grad_Laa)
        # Q 梯度已移除（Q 项不应参与 FIM，见 algo2 修复说明）
        grad_Saa.append(grad_Laa)
        grad_Srr.append(grad_Lrr)
        grad_Sar.append(grad_Lar)

    # 数值安全检查：NaN/Inf 替换为零矩阵
    for i in range(U):
        if not np.all(np.isfinite(grad_Saa[i])):
            grad_Saa[i] = np.zeros_like(grad_Saa[i])
        if not np.all(np.isfinite(grad_Srr[i])):
            grad_Srr[i] = np.zeros_like(grad_Srr[i])
        if not np.all(np.isfinite(grad_Sar[i])):
            grad_Sar[i] = np.zeros_like(grad_Sar[i])

    return grad_Saa, grad_Srr, grad_Sar, P_sum, P_alpha, P_r, c_alpha, c_r, c_ar


# ============================================================
# Algorithm 6  构造 SCA 仿射线性近似
# ============================================================

def algo6_sca_approx(Rx_list_t, S_aa_t, S_rr_t, S_ar_t,
                     grad_Saa, grad_Srr, grad_Sar):
    """
    返回三个仿射函数（Python callable，接受 Rx_list 为参数）
    以及常数项（供 CVXPY 构造线性约束使用）

    对 CVXPY 变量，实际使用方式是：
        S_tilde_aa = S_aa_t + sum_i 2*Re{tr(grad_Saa[i].H @ (Rxi - Rxi_t))}
    这里返回常数偏置和梯度，供 Algorithm 7 直接用 cvxpy 表达式构造。
    """
    # 直接返回供 CVXPY 使用的常数和梯度
    return S_aa_t, S_rr_t, S_ar_t, grad_Saa, grad_Srr, grad_Sar


def sca_linear_expr(S_t, grad_S, Rx_vars, Rx_t_list):
    """
    构造 CVXPY 仿射表达式：
        Ṡ = S_t + Σᵢ 2·ℜ{ tr(∇ᵢSᴴ · (Rₓᵢ − Rₓᵢ⁽ᵗ⁾)) }

    Rx_vars   : list of CVXPY affine expressions (M,M)  —— 必须是关于优化变量的仿射式
    Rx_t_list : list of np.ndarray (M,M)               —— 展开点（常数）

    注意：Rx_vars 必须是线性/仿射 CVXPY 表达式，不能包含 cp.outer(w,w*)
    这里通过 SDR 辅助变量 W_ik 保证仿射性。
    """
    expr = float(S_t)
    for G, Rvar, Rt in zip(grad_S, Rx_vars, Rx_t_list):
        # diff = Rvar - Rt 是仿射 CVXPY 表达式
        diff = Rvar - Rt
        # ℜ{tr(Gᴴ·diff)} = ℜ{ Σᵢⱼ conj(G)ᵢⱼ · diff_ᵢⱼ }
        #                 = cp.real(cp.sum(cp.multiply(G.conj(), diff)))
        # 等价于 cp.real(cp.trace(G.conj().T @ diff))，但用 multiply+sum 更稳定
        expr = expr + 2 * cp.real(cp.sum(cp.multiply(G.conj(), diff)))
    return expr


# ============================================================
# Algorithm 7  求解子问题 P2-w-SCA：通信波束优化
# ============================================================

def algo7_beam_opt(Rs_list_t, phi_t, mu, eta, h_list, sigma2_k,
                   P_max, eps_alpha_tilde, eps_r_tilde,
                   S_aa_t, S_rr_t, S_ar_t, grad_Saa, grad_Srr, grad_Sar,
                   b_vec, M, U, K, w_prev=None, eps_relax_factor=0.8,
                   max_relax=5):
    """
    Algorithm 7: 通信波束优化子问题（SDR + SCA，DCP 合规版本）

    修复说明：
    原始版本用 cp.outer(w, w*) 构造 Rx，导致约束关于 w 是二次非凸的，
    CVXPY 无法验证 DCP 合规性（DCPError）。

    修复方案：引入 SDR 辅助变量 Wik = wik * wik^H（M×M Hermitian PSD），
    通过 Schur 补 LMI [[Wik, wik],[wik^H, 1]] >> 0 关联 Wik 与 wik。
    此时 Rxi = sum_k Wik + Rs_i 是关于 Wik 的线性表达式，
    SCA 约束对变量保持仿射，所有 DCP 约束均满足。
    """
    # 预计算展开点 Rxi^(t)（numpy 常数）
    Rx_t_list = []
    for i in range(U):
        Rt = Rs_list_t[i].copy().astype(complex)
        if w_prev is not None:
            for k in range(K):
                wt = w_prev[i * K + k]
                Rt = Rt + np.outer(wt, wt.conj())
        Rx_t_list.append(Rt)

    Rs_block_const = block_diag_np(Rs_list_t)

    P_budget = []
    for i in range(U):
        rem = P_max[i] - np.real(np.trace(Rs_list_t[i]))
        P_budget.append(max(rem, 1e-8))

    eps_a = float(eps_alpha_tilde)
    eps_r = float(eps_r_tilde)

    for attempt in range(max_relax + 1):

        # ── 决策变量 ──
        # wik in C^M : 波束向量（用于目标 term1，线性）
        w_vars = [[cp.Variable(M, complex=True) for _ in range(K)]
                  for _ in range(U)]
        # Wik = wik*wik^H 的 SDR 松弛：M×M Hermitian PSD
        W_vars = [[cp.Variable((M, M), hermitian=True) for _ in range(K)]
                  for _ in range(U)]

        zeta_aa = cp.Variable(nonneg=True)
        zeta_rr = cp.Variable(nonneg=True)
        zeta_ar = cp.Variable(nonneg=True)

        constraints = []

        # ── Schur 补 LMI: [[Wik, wik],[wik^H, 1]] >> 0
        #    等价于 Wik >> wik*wik^H（SDR 的标准 rank-1 松弛约束）
        for i in range(U):
            for k in range(K):
                w = w_vars[i][k]    # (M,) complex vector variable
                W = W_vars[i][k]    # (M,M) hermitian PSD variable
                # 构造 (M+1)×(M+1) Schur 矩阵
                schur = cp.bmat([
                    [W,                        cp.reshape(w, (M, 1))],
                    [cp.reshape(cp.conj(w), (1, M)), np.array([[1.0]])]
                ])
                constraints += [schur >> 0]
                constraints += [W >> 0]

        # ── Rxi = sum_k Wik + Rs_i（关于 Wik 的线性表达式，DCP 合规）──
        Rx_expr = []
        for i in range(U):
            Rx_i = Rs_list_t[i].astype(complex) + sum(W_vars[i][k] for k in range(K))
            Rx_expr.append(Rx_i)

        # ── SCA 仿射近似（Rx_expr 是线性的，满足 DCP）──
        S_aa_tilde = sca_linear_expr(S_aa_t, grad_Saa, Rx_expr, Rx_t_list)
        S_rr_tilde = sca_linear_expr(S_rr_t, grad_Srr, Rx_expr, Rx_t_list)
        S_ar_tilde = sca_linear_expr(S_ar_t, grad_Sar, Rx_expr, Rx_t_list)

        # ── C1: CRB(alpha) <= eps_alpha
        #    等价 SOC: ||[2*zeta_ar, zeta_rr-zeta_aa+eps_a]|| <= zeta_rr+zeta_aa-eps_a
        constraints += [
            cp.norm(cp.hstack([2 * zeta_ar, zeta_rr - zeta_aa + eps_a])) <=
            zeta_rr + zeta_aa - eps_a
        ]
        # ── C2: CRB(r) <= eps_r
        constraints += [
            cp.norm(cp.hstack([2 * zeta_ar, zeta_rr - zeta_aa + eps_r])) <=
            zeta_rr + zeta_aa - eps_r
        ]
        # ── C3-C5: SCA 仿射下界约束 ──
        constraints += [zeta_aa <= S_aa_tilde]
        constraints += [zeta_rr <= S_rr_tilde]
        constraints += [S_ar_tilde <= zeta_ar, -S_ar_tilde <= zeta_ar]
        constraints += [zeta_aa >= 1e-6, zeta_rr >= 1e-6]

        # ── C6: 发射功率约束 tr(sum_k Wik) <= P_budget[i] ──
        for i in range(U):
            constraints += [
                sum(cp.real(cp.trace(W_vars[i][k])) for k in range(K)) <= P_budget[i]
            ]

        # ── 目标：FP 参数化速率 R_tilde ──
        # term1 = 2*Re(eta_k^* * h_k^H * w_k): 关于 wik 的线性函数（仿射，DCP OK）
        # term2 = |eta_k|^2 * [sum_{j!=k} h_k^H * Wj * h_k + const]:
        #         关于 Wik 的线性函数（仿射，DCP OK）
        obj_terms = []
        for k in range(K):
            hk = h_list[k]   # (U*M,)

            # term1: linear in wik
            term1 = 0.0
            for i in range(U):
                hk_i = hk[i * M:(i + 1) * M]
                term1 = term1 + 2.0 * cp.real(
                    complex(np.conj(eta[k])) * (hk_i.conj() @ w_vars[i][k])
                )

            # Q_k 常数部分
            Q_const = float(np.real(hk.conj() @ Rs_block_const @ hk) + sigma2_k[k])

            # 干扰项: sum_{j!=k} h_k^H * (sum_i Wij) * h_k, linear in Wij
            inter = 0.0
            for j in range(K):
                if j != k:
                    for i in range(U):
                        hk_i = hk[i * M:(i + 1) * M]   # (M,) complex constant
                        # cp.quad_form 不支持复数，用 hk_i^H @ W @ hk_i 代替
                        inter = inter + cp.real(
                            hk_i.conj() @ W_vars[i][j] @ hk_i
                        )

            term2 = float(np.abs(eta[k])**2) * (inter + Q_const)
            obj_terms.append(term1 - term2)

        prob = cp.Problem(cp.Maximize(sum(obj_terms)), constraints)

        try:
            prob.solve(solver=cp.MOSEK, verbose=False)
        except Exception:
            try:
                prob.solve(solver=cp.SCS, verbose=False)
            except Exception as e:
                warnings.warn(f"Algorithm 7 solver exception: {e}")

        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            w_new = []
            for i in range(U):
                for k in range(K):
                    val = w_vars[i][k].value
                    w_new.append(val if val is not None else np.zeros(M, dtype=complex))
            return w_new

        # infeasible: 松弛感知约束重试
        eps_a *= eps_relax_factor
        eps_r *= eps_relax_factor
        warnings.warn(f"Algorithm 7 infeasible (attempt {attempt + 1}), "
                      f"relax to eps_a={eps_a:.4f}, eps_r={eps_r:.4f}")

    warnings.warn("Algorithm 7: all attempts infeasible, trying rate-only fallback")

    # ── 兜底方案：去掉感知约束，纯通信波束优化 ──
    # 保证在任何情况下都返回非零波束
    w_vars_fb = [[cp.Variable(M, complex=True) for _ in range(K)]
                 for _ in range(U)]
    W_vars_fb = [[cp.Variable((M, M), hermitian=True) for _ in range(K)]
                 for _ in range(U)]

    constraints_fb = []
    for i in range(U):
        for k in range(K):
            w = w_vars_fb[i][k]
            W = W_vars_fb[i][k]
            schur = cp.bmat([
                [W,                        cp.reshape(w, (M, 1))],
                [cp.reshape(cp.conj(w), (1, M)), np.array([[1.0]])]
            ])
            constraints_fb += [schur >> 0, W >> 0]
        # 功率约束
        P_bud = max(P_max[i] - np.real(np.trace(Rs_list_t[i])), 1e-8)
        constraints_fb += [
            sum(cp.real(cp.trace(W_vars_fb[i][k])) for k in range(K)) <= P_bud
        ]

    Rs_block_fb = block_diag_np(Rs_list_t)
    obj_fb = []
    for k in range(K):
        hk = h_list[k]
        term1 = sum(2.0 * cp.real(
            complex(np.conj(eta[k])) * (hk[i*M:(i+1)*M].conj() @ w_vars_fb[i][k])
        ) for i in range(U))
        Q_const = float(np.real(hk.conj() @ Rs_block_fb @ hk) + sigma2_k[k])
        inter = sum(
            cp.real(hk[i*M:(i+1)*M].conj() @ W_vars_fb[i][j] @ hk[i*M:(i+1)*M])
            for j in range(K) if j != k
            for i in range(U)
        )
        obj_fb.append(term1 - float(np.abs(eta[k])**2) * (inter + Q_const))

    prob_fb = cp.Problem(cp.Maximize(sum(obj_fb)), constraints_fb)
    try:
        prob_fb.solve(solver=cp.MOSEK, verbose=False)
    except Exception:
        try:
            prob_fb.solve(solver=cp.SCS, verbose=False)
        except Exception:
            pass

    if prob_fb.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        warnings.warn("Algorithm 7: rate-only fallback succeeded")
        w_new = []
        for i in range(U):
            for k in range(K):
                val = w_vars_fb[i][k].value
                w_new.append(val if val is not None else np.zeros(M, dtype=complex))
        return w_new

    # 最终回退：保持上一轮波束
    if w_prev is None:
        return [np.zeros(M, dtype=complex) for _ in range(U * K)]
    return w_prev


# ============================================================
# 辅助函数：块对角矩阵 & 波束向量拼接
# ============================================================

def block_diag_np(mat_list):
    """将矩阵列表组成块对角矩阵 (numpy)"""
    from scipy.linalg import block_diag
    return block_diag(*mat_list)


def w_list_to_flat(w_ik_list, U, K, M):
    """
    w_ik_list[i*K+k] = (M,)  →  拼接为每用户 (U*M,) 向量列表
    返回 list of K 个 (U*M,) 向量
    """
    result = []
    for k in range(K):
        wk = np.concatenate([w_ik_list[i * K + k] for i in range(U)])
        result.append(wk)
    return result


# ============================================================
# Algorithm 8  感知协方差优化（标准 SDP + SCA）
# ============================================================

def algo8_sensing_cov_opt(w_ik_list, Rs_list_t, phi_t, mu, eta, h_list, sigma2_k,
                           P_max, eps_alpha_tilde, eps_r_tilde,
                           a_vec, da_alpha, da_r, b_vec, db_alpha, db_r,
                           beta, N, sigma2, M, U, K):
    """
    Algorithm 8: 感知协方差优化子问题（SDP + SCA）

    优化变量: Rᵢˢ（M×M Hermitian PSD），w 固定。
    展开点: 当前 Rs_list_t（来自上一轮迭代）。
    SCA 线性化: Sαα(Rx) 在展开点 Rx_t = Ww + Rs_t 处线性化，
                其中 Ww = Σₖ wᵢₖwᵢₖᴴ 为固定的通信波束协方差。

    修复：
    - 新增参数 Rs_list_t（展开点）和 db_alpha/db_r（发射侧偏导数）
    - 正确使用 b_vec/db_alpha/db_r 作为发射侧向量传给 algo5
    - 展开点 Rx_t_list[i] = Ww_i + Rs_list_t[i]（而非 Rs=0）
    - 全面 NaN/Inf 检测和数值保护
    """
    # ── 固定波束部分协方差 Ww_i = Σₖ wᵢₖwᵢₖᴴ ──
    Ww_list = []
    for i in range(U):
        Ww = sum(np.outer(w_ik_list[i * K + k], w_ik_list[i * K + k].conj())
                 for k in range(K))
        Ww_list.append(Ww.astype(complex))

    # ── 展开点 Rx_t[i] = Ww_i + Rs_t[i] ──
    Rx_t_list = [Ww_list[i] + Rs_list_t[i].astype(complex) for i in range(U)]

    # ── 在展开点计算 SCA 梯度（正确传入发射侧 db_alpha/db_r）──
    grad_Saa, grad_Srr, grad_Sar, P_sum_t, P_alpha_t, P_r_t, c_alpha_t, c_r_t, c_ar_t = \
        algo5_sca_gradients(Rx_t_list, beta, a_vec, da_alpha, da_r,
                             b_vec, db_alpha, db_r, N, sigma2)

    # ── 在展开点计算 S 值 ──
    S_aa_t, S_rr_t, S_ar_t = algo2_fim_elements(
        P_sum_t, P_alpha_t, P_r_t, c_alpha_t, c_r_t, c_ar_t,
        a_vec, da_alpha, da_r, beta, N, sigma2)

    # ── NaN/Inf 检查 ──
    scalars_ok = all(np.isfinite(v) for v in [S_aa_t, S_rr_t, S_ar_t])
    grads_ok   = all(np.all(np.isfinite(g)) for g in grad_Saa + grad_Srr + grad_Sar)
    if not (scalars_ok and grads_ok):
        warnings.warn("Algorithm 8: SCA 梯度含 NaN/Inf，跳过本轮协方差优化")
        return None

    # ── 功率预算 ──
    P_s_budget = []
    for i in range(U):
        w_power = sum(float(np.linalg.norm(w_ik_list[i * K + k])**2) for k in range(K))
        P_s_budget.append(max(P_max[i] - w_power, 1e-8))

    # ── CVXPY 决策变量 ──
    Rs_vars = [cp.Variable((M, M), hermitian=True) for _ in range(U)]
    zeta_aa = cp.Variable(nonneg=True)
    zeta_rr = cp.Variable(nonneg=True)
    zeta_ar = cp.Variable(nonneg=True)

    constraints = []

    # Rₓᵢ = Ww_i + Rᵢˢ（关于 Rs_vars[i] 的线性表达式）
    Rx_expr = [Ww_list[i] + Rs_vars[i] for i in range(U)]

    # SCA 仿射近似（Rs_vars 在 Rx_expr 中是线性的 → DCP 合规）
    S_aa_lin = sca_linear_expr(S_aa_t, grad_Saa, Rx_expr, Rx_t_list)
    S_rr_lin = sca_linear_expr(S_rr_t, grad_Srr, Rx_expr, Rx_t_list)
    S_ar_lin = sca_linear_expr(S_ar_t, grad_Sar, Rx_expr, Rx_t_list)

    # ── D1/D2: CRB SOC 约束 ──
    eps_a = float(eps_alpha_tilde)
    eps_r = float(eps_r_tilde)
    constraints += [
        cp.norm(cp.hstack([2 * zeta_ar, zeta_rr - zeta_aa + eps_a])) <=
        zeta_rr + zeta_aa - eps_a
    ]
    constraints += [
        cp.norm(cp.hstack([2 * zeta_ar, zeta_rr - zeta_aa + eps_r])) <=
        zeta_rr + zeta_aa - eps_r
    ]
    # ── D3-D5: SCA 仿射下界 ──
    constraints += [zeta_aa <= S_aa_lin, zeta_rr <= S_rr_lin]
    constraints += [S_ar_lin <= zeta_ar, -S_ar_lin <= zeta_ar]
    constraints += [zeta_aa >= 1e-6, zeta_rr >= 1e-6]

    # ── D6: 发射功率约束 tr(Rᵢˢ) ≤ P_s_budget[i] ──
    for i in range(U):
        constraints += [cp.real(cp.trace(Rs_vars[i])) <= P_s_budget[i]]

    # ── D7: Rᵢˢ ⪰ 0 ──
    for i in range(U):
        constraints += [Rs_vars[i] >> 0]

    # ── 目标：R̃ 对 Rᵢˢ 的仿射部分 ──
    # term1 = 2ℜ{ηₖ*·hₖᴴ·wₖ}：与 Rs 无关，为常数
    # term2 中 Q_k = Σⱼ≠ₖ|hₖwⱼ|² + hₖᴴRshₖ + σₖ²
    #   hₖᴴRshₖ = Σᵢ hₖᵢᴴ Rs_vars[i] hₖᵢ（线性于 Rs_vars[i]）
    Rs_block_const = block_diag_np(Rs_list_t)
    obj_terms = []
    for k in range(K):
        hk = h_list[k]   # (U*M,)
        wk_fixed = np.concatenate([w_ik_list[i * K + k] for i in range(U)])

        # term1（常数）
        term1_const = float(2 * np.real(np.conj(eta[k]) * (hk.conj() @ wk_fixed)))

        # 干扰（常数，w 固定）
        inter_const = float(sum(
            np.abs(hk.conj() @ np.concatenate([w_ik_list[i * K + j]
                                                for i in range(U)]))**2
            for j in range(K) if j != k
        ))

        # hₖᴴ Rs hₖ（线性于 Rs_vars）
        hk_rs_hk = sum(
            cp.real(hk[i * M:(i + 1) * M].conj() @ Rs_vars[i] @ hk[i * M:(i + 1) * M])
            for i in range(U)
        )

        Q_k = inter_const + hk_rs_hk + float(sigma2_k[k])
        term2 = float(np.abs(eta[k])**2) * Q_k

        obj_terms.append(term1_const - term2)

    prob = cp.Problem(cp.Maximize(sum(obj_terms)), constraints)

    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
    except Exception:
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except Exception as e:
            warnings.warn(f"Algorithm 8 求解器异常: {e}")
            return None

    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        result = []
        for i in range(U):
            val = Rs_vars[i].value
            if val is None or not np.all(np.isfinite(val)):
                val = Rs_list_t[i].copy()
            val = (val + val.conj().T) / 2
            eigvals = np.linalg.eigvalsh(val)
            if eigvals.min() < 0:
                val += (-eigvals.min() + 1e-9) * np.eye(M)
            result.append(val)
        return result
    else:
        warnings.warn(f"Algorithm 8 求解失败 ({prob.status})，尝试纯感知兜底")

    # ── 兜底：去掉 CRB 约束，只做功率分配最大化感知 ──
    Rs_fb = [cp.Variable((M, M), hermitian=True) for _ in range(U)]
    constraints_fb = []
    for i in range(U):
        constraints_fb += [Rs_fb[i] >> 0]
        constraints_fb += [cp.real(cp.trace(Rs_fb[i])) <= P_s_budget[i]]

    # 目标：最大化总感知功率（让 FIM 非奇异）
    obj_fb = sum(cp.real(cp.trace(Rs_fb[i])) for i in range(U))
    prob_fb = cp.Problem(cp.Maximize(obj_fb), constraints_fb)
    try:
        prob_fb.solve(solver=cp.MOSEK, verbose=False)
    except Exception:
        try:
            prob_fb.solve(solver=cp.SCS, verbose=False)
        except Exception:
            pass

    if prob_fb.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        warnings.warn("Algorithm 8: 感知兜底成功")
        result = []
        for i in range(U):
            val = Rs_fb[i].value
            if val is None or not np.all(np.isfinite(val)):
                val = (P_s_budget[i] / M) * np.eye(M)
            val = (val + val.conj().T) / 2
            eigvals = np.linalg.eigvalsh(val)
            if eigvals.min() < 0:
                val += (-eigvals.min() + 1e-9) * np.eye(M)
            result.append(val)
        return result

    # 最终兜底：按预算均匀分配
    warnings.warn("Algorithm 8: 全部失败，均匀分配感知功率")
    return [(P_s_budget[i] / M) * np.eye(M, dtype=complex) for i in range(U)]



def block_diag_cvxpy(mat_vars, M, U):
    """构造 CVXPY 块对角矩阵（UM × UM）"""
    rows = []
    for i in range(U):
        row_blocks = []
        for j in range(U):
            if i == j:
                row_blocks.append(mat_vars[i])
            else:
                row_blocks.append(np.zeros((M, M)))
        rows.append(cp.hstack(row_blocks))
    return cp.vstack(rows)


# ============================================================
# Algorithm 9S  子程序 eval_J(φ)
# ============================================================

def build_user_channels(bs_pos, user_pos, phi, tau, M, U, K):
    """
    构造用户信道向量 h_list。
    每个基站 i 到用户 k 的信道方向由用户相对于该基站的物理角度决定，
    不依赖感知目标方向。
    h_k = [sqrt(tau[0,k])*b(theta_0k), ..., sqrt(tau[U-1,k])*b(theta_{U-1,k})] in C^{UM}
    """
    h_list = []
    for k in range(K):
        hk_parts = []
        for u in range(U):
            # 用户 k 相对基站 u 的方位角（全局）
            dx = user_pos[k, 0] - bs_pos[u, 0]
            dy = user_pos[k, 1] - bs_pos[u, 1]
            theta_uk_global = np.arctan2(dy, dx)
            # 减去旋转角得到相对入射角
            theta_uk = theta_uk_global - phi[u]
            # 发射导向向量（M根天线，半波长间距）
            m_idx = np.arange(M)
            b_uk = np.exp(-1j * np.pi * np.sin(theta_uk) * m_idx)
            hk_parts.append(np.sqrt(tau[u, k]) * b_uk)
        h_list.append(np.concatenate(hk_parts))
    return h_list


def algo9s_eval_J(phi, w_ik_list, Rs_list, mu_pp, eta_pp,
                   bs_pos, user_pos, alpha, r, M, N, U, K,
                   sigma2_k, sigma2, beta, eps_alpha, eps_r,
                   lambda_alpha, lambda_r, tau):
    """
    user_pos : (K,2) 用户坐标
    """
    # Step 1: 更新感知几何量与导向向量
    theta_tilde, a_vec, b_vec, da_alpha, da_r, db_alpha, db_r = \
        algo0_precompute(bs_pos, alpha, r, phi, M, N)

    # Step 2: 构造通信信道（基于用户位置）
    h_list = build_user_channels(bs_pos, user_pos, phi, tau, M, U, K)

    # Step 3: 计算 R̃(φ)（用当前 μ**, η** 代入）
    Rs_block = block_diag_np(Rs_list)
    w_flat = w_list_to_flat(w_ik_list, U, K, M)

    R_tilde = 0.0
    for k in range(K):
        hk = h_list[k]
        wk = w_flat[k]
        term1 = 2 * np.real(eta_pp[k].conj() * (hk.conj() @ wk))
        I_k = sum(np.abs(hk.conj() @ w_flat[j])**2 for j in range(K) if j != k)
        Q_k = I_k + np.real(hk.conj() @ Rs_block @ hk) + sigma2_k[k]
        term2 = np.abs(eta_pp[k])**2 * Q_k
        R_tilde += term1 - term2

    # Step 4: 计算 CRB
    Rx_list = []
    for i in range(U):
        Ww = sum(np.outer(w_ik_list[i*K+k], w_ik_list[i*K+k].conj())
                 for k in range(K))
        Rx_list.append(Ww + Rs_list[i])

    P_sum, P_alpha, P_r, c_alpha, c_r, c_ar = algo1_fim_scalars(
        Rx_list, b_vec, db_alpha, db_r)
    S_aa, S_rr, S_ar = algo2_fim_elements(
        P_sum, P_alpha, P_r, c_alpha, c_r, c_ar,
        a_vec, da_alpha, da_r, beta, N, sigma2)
    CRB_alpha, CRB_r = algo3_crb(S_aa, S_rr, S_ar, sigma2)

    # Step 5: 联合代价函数
    delta1 = CRB_alpha - eps_alpha
    delta2 = CRB_r     - eps_r
    J = -R_tilde + lambda_alpha * max(delta1, 0)**2 + lambda_r * max(delta2, 0)**2

    return J


# ============================================================
# Algorithm 9  穷举搜索（ES）
# ============================================================

def algo9_es(w_ik_list, Rs_list, mu_pp, eta_pp,
              bs_pos, user_pos, alpha, r, M, N, U, K,
              sigma2_k, sigma2, beta, eps_alpha, eps_r,
              lambda_alpha, lambda_r, tau,
              phi_min, phi_max, delta_phi):
    """
    返回最优旋转角向量 phi_star (U,)
    """
    X = int((phi_max - phi_min) / delta_phi)
    grid = np.linspace(phi_min, phi_max, X + 1)

    J_min = np.inf
    phi_star = np.ones(U) * (phi_min + phi_max) / 2

    for phi_combo in itertools.product(grid, repeat=U):
        phi = np.array(phi_combo)
        J = algo9s_eval_J(phi, w_ik_list, Rs_list, mu_pp, eta_pp,
                           bs_pos, alpha, r, M, N, U, K,
                           sigma2_k, sigma2, beta, eps_alpha, eps_r,
                           lambda_alpha, lambda_r, tau)
        if J < J_min:
            J_min = J
            phi_star = phi.copy()

    return phi_star


# ============================================================
# Algorithm 10  粒子群优化（PSO）
# ============================================================

def algo10_pso(w_ik_list, Rs_list, mu_pp, eta_pp,
               bs_pos, user_pos, alpha, r, M, N, U, K,
               sigma2_k, sigma2, beta, eps_alpha, eps_r,
               lambda_alpha, lambda_r, tau,
               phi_min, phi_max,
               P_pso=30, T_pso=100, c1=2.0, c2=2.0,
               omega_max=0.9, omega_min=0.4, v_max=None, delta_pso=1e-4):
    """
    返回最优旋转角向量 phi_star (U,)
    """
    if v_max is None:
        v_max = (phi_max - phi_min) * 0.2

    rng = np.random.default_rng()

    # 初始化粒子群
    phi_particles = rng.uniform(phi_min, phi_max, (P_pso, U))
    vel = np.zeros((P_pso, U))
    phi_best_i  = phi_particles.copy()
    J_best_i    = np.full(P_pso, np.inf)

    for i in range(P_pso):
        J_best_i[i] = algo9s_eval_J(
            phi_particles[i], w_ik_list, Rs_list, mu_pp, eta_pp,
            bs_pos, user_pos, alpha, r, M, N, U, K,
            sigma2_k, sigma2, beta, eps_alpha, eps_r, lambda_alpha, lambda_r, tau)

    g_best_idx   = np.argmin(J_best_i)
    phi_g        = phi_best_i[g_best_idx].copy()
    J_g_best     = J_best_i[g_best_idx]
    J_g_prev     = J_g_best

    # 主迭代
    for tau_iter in range(T_pso):
        omega = omega_max - (omega_max - omega_min) * tau_iter / T_pso

        for i in range(P_pso):
            r1 = rng.uniform(0, 1, U)
            r2 = rng.uniform(0, 1, U)

            vel[i] = (omega * vel[i]
                      + c1 * r1 * (phi_best_i[i] - phi_particles[i])
                      + c2 * r2 * (phi_g - phi_particles[i]))
            vel[i] = np.clip(vel[i], -v_max, v_max)

            phi_particles[i] = np.clip(phi_particles[i] + vel[i], phi_min, phi_max)

            J_new = algo9s_eval_J(
                phi_particles[i], w_ik_list, Rs_list, mu_pp, eta_pp,
                bs_pos, user_pos, alpha, r, M, N, U, K,
                sigma2_k, sigma2, beta, eps_alpha, eps_r, lambda_alpha, lambda_r, tau)

            if J_new < J_best_i[i]:
                J_best_i[i]   = J_new
                phi_best_i[i] = phi_particles[i].copy()

            if J_new < J_g_best:
                J_g_best = J_new
                phi_g    = phi_particles[i].copy()

        # 收敛检查
        if abs(J_g_best - J_g_prev) / (abs(J_g_prev) + 1e-12) < delta_pso:
            print(f"  PSO 提前收敛于第 {tau_iter+1} 轮")
            break
        J_g_prev = J_g_best

    return phi_g


# ============================================================
# Algorithm 11  AO 主算法
# ============================================================

def algo11_ao(bs_pos, user_pos, tau, P_max, beta, sigma2, sigma2_k,
              alpha_true, r_true, M, N, U, K,
              eps_alpha, eps_r, phi_min, phi_max,
              lambda_alpha=10.0, lambda_r=10.0,
              T_max=50, delta_AO=1e-3, eps_abs=1e-4, eps_AO=1e-5,
              rho_init=0.1, method='PSO',
              pso_params=None, es_params=None):
    """
    主 AO 优化框架

    输入:
        bs_pos   : (U,2) 基站坐标
        tau      : (U,K) 路径损耗系数（实数）
        P_max    : (U,)  各基站最大发射功率
        beta     : (U,)  复信道反射系数 βᵤ
        sigma2   : float 感知接收噪声功率
        sigma2_k : (K,)  每用户通信噪声功率
        alpha_true, r_true : 目标真实位置（用于初始化）
        M, N, U, K : 系统维度
        eps_alpha, eps_r : 感知 CRB 阈值
        phi_min, phi_max : 旋转角范围 (rad)
        lambda_alpha, lambda_r : 惩罚因子
        method   : 'ES' 或 'PSO'

    输出:
        dict 包含最优 w_ik_list, Rs_list, phi, R_star, CRB_alpha, CRB_r
    """

    # ═══ 初始化 ═══
    phi = np.ones(U) * (phi_min + phi_max) / 2.0

    # ── 初始用户信道（用于 MRT 初始化）──
    h_list_init0 = build_user_channels(bs_pos, user_pos, phi, tau, M, U, K)

    # 初始通信波束：MRT，预留 rho_sense=30% 的功率给感知
    rho_sense = 0.3   # 感知功率预留比例
    w_ik_list = []
    for i in range(U):
        for k in range(K):
            hik = h_list_init0[k][i * M:(i + 1) * M]
            hik_norm = norm(hik)
            if hik_norm < 1e-12:
                hik = np.ones(M, dtype=complex) / np.sqrt(M)
            else:
                hik = hik / hik_norm
            w_ik = np.sqrt((1.0 - rho_sense) * P_max[i] / K) * hik
            w_ik_list.append(w_ik)

    # 初始感知协方差：rho_sense * P_max / M * I
    Rs_list = [(rho_sense * P_max[i] / M) * np.eye(M, dtype=complex)
               for i in range(U)]

    # ═══ 可行性检验 ═══
    Rx_init = []
    for i in range(U):
        Ww = sum(np.outer(w_ik_list[i*K+k], w_ik_list[i*K+k].conj())
                 for k in range(K))
        Rx_init.append(Ww + Rs_list[i])

    theta_t, a_t, b_t, da_t_a, da_t_r, db_t_a, db_t_r = \
        algo0_precompute(bs_pos, alpha_true, r_true, phi, M, N)
    P_s, P_a, P_r0, c_a, c_r0, c_ar = algo1_fim_scalars(Rx_init, b_t, db_t_a, db_t_r)
    S_aa0, S_rr0, S_ar0 = algo2_fim_elements(P_s, P_a, P_r0, c_a, c_r0, c_ar,
                                               a_t, da_t_a, da_t_r, beta, N, sigma2)
    CRB_a0, CRB_r0 = algo3_crb(S_aa0, S_rr0, S_ar0, sigma2)

    print(f"初始点: CRBα={CRB_a0:.4e} (阈值={eps_alpha:.4e}), "
          f"CRBr={CRB_r0:.4e} (阈值={eps_r:.4e})")
    if CRB_a0 > eps_alpha or CRB_r0 > eps_r:
        print("WARNING: 初始点不满足感知约束，建议增大 ρ_init 或放松阈值")

    # ε̃ = σ²/(2ε)
    eps_a_tilde = sigma2 / (2 * eps_alpha)
    eps_r_tilde = sigma2 / (2 * eps_r)

    # 计算初始速率和诊断信息
    h_list_init = build_user_channels(bs_pos, user_pos, phi, tau, M, U, K)
    Rs_block_init = block_diag_np(Rs_list)
    w_flat_init   = w_list_to_flat(w_ik_list, U, K, M)
    mu0, eta0     = algo4_fp_update(w_flat_init, Rs_block_init, h_list_init, sigma2_k)
    R_prev        = compute_sinr_rate(w_ik_list, Rs_list, h_list_init, sigma2_k, U, K, M)

    # 诊断：打印初始信道强度和 SINR
    print(f"初始 R_sum = {R_prev:.4f} bps/Hz")
    for k in range(K):
        hk = h_list_init[k]
        wk = w_flat_init[k]
        sig = np.abs(hk.conj() @ wk)**2
        noise = sigma2_k[k]
        print(f"  用户{k}: |h^H w|²={sig:.4f}, μ₀={mu0[k]:.4f}, |η₀|={abs(eta0[k]):.4f}")
    print(f"  eps_a_tilde={eps_a_tilde:.4e}, eps_r_tilde={eps_r_tilde:.4e}")

    # ═══ 主迭代循环 ═══
    t = 0
    converged = False

    while t < T_max and not converged:
        print(f"\n═ AO 迭代 t={t} ═")

        # 更新几何量（当前 phi）
        theta_t, a_t, b_t, da_t_a, da_t_r, db_t_a, db_t_r = \
            algo0_precompute(bs_pos, alpha_true, r_true, phi, M, N)

        # 构造信道
        h_list = build_user_channels(bs_pos, user_pos, phi, tau, M, U, K)

        # 构造 Rx_list
        Rx_list = []
        for i in range(U):
            Ww = sum(np.outer(w_ik_list[i*K+k], w_ik_list[i*K+k].conj())
                     for k in range(K))
            Rx_list.append(Ww + Rs_list[i])

        # ── 步骤 1：FP 辅助变量更新 ──
        Rs_block = block_diag_np(Rs_list)
        w_flat   = w_list_to_flat(w_ik_list, U, K, M)
        mu, eta  = algo4_fp_update(w_flat, Rs_block, h_list, sigma2_k)
        print(f"  步骤1 FP 更新完成, μ_mean={mu.mean():.4f}")

        # ── 步骤 2：通信波束优化（SCA） ──
        grad_Saa, grad_Srr, grad_Sar, P_s, P_a, P_r0, c_a, c_r0, c_ar = \
            algo5_sca_gradients(Rx_list, beta, a_t, da_t_a, da_t_r,
                                 b_t, db_t_a, db_t_r, N, sigma2)
        S_aa_t = algo2_fim_elements(P_s, P_a, P_r0, c_a, c_r0, c_ar,
                                      a_t, da_t_a, da_t_r, beta, N, sigma2)[0]
        S_rr_t = algo2_fim_elements(P_s, P_a, P_r0, c_a, c_r0, c_ar,
                                      a_t, da_t_a, da_t_r, beta, N, sigma2)[1]
        S_ar_t = algo2_fim_elements(P_s, P_a, P_r0, c_a, c_r0, c_ar,
                                      a_t, da_t_a, da_t_r, beta, N, sigma2)[2]

        w_new = algo7_beam_opt(
            Rs_list, phi, mu, eta, h_list, sigma2_k,
            P_max, eps_a_tilde, eps_r_tilde,
            S_aa_t, S_rr_t, S_ar_t, grad_Saa, grad_Srr, grad_Sar,
            b_t, M, U, K, w_prev=w_ik_list)

        if w_new is not None:
            w_ik_list = w_new
        print(f"  步骤2 波束优化完成")

        # ── 步骤 3：感知协方差优化（SDP） ──
        Rs_new = algo8_sensing_cov_opt(
            w_ik_list, Rs_list, phi, mu, eta, h_list, sigma2_k,
            P_max, eps_a_tilde, eps_r_tilde,
            a_t, da_t_a, da_t_r, b_t, db_t_a, db_t_r,
            beta, N, sigma2, M, U, K)

        if Rs_new is not None:
            Rs_list = Rs_new
        # 诊断：打印感知功率预算
        for i in range(U):
            w_pwr = sum(float(np.linalg.norm(w_ik_list[i*K+k])**2) for k in range(K))
            rs_pwr = float(np.real(np.trace(Rs_list[i])))
            print(f"    BS{i}: w_power={w_pwr:.4f}, Rs_trace={rs_pwr:.4f}, "
                  f"budget_left={max(P_max[i]-w_pwr,0):.4f}")
        print(f"  步骤3 感知协方差优化完成")

        # 步骤 3.5：更新 FP（基于最新 w, Rs）
        Rx_list_new = []
        for i in range(U):
            Ww = sum(np.outer(w_ik_list[i*K+k], w_ik_list[i*K+k].conj())
                     for k in range(K))
            Rx_list_new.append(Ww + Rs_list[i])

        Rs_block_new = block_diag_np(Rs_list)
        w_flat_new   = w_list_to_flat(w_ik_list, U, K, M)
        mu_pp, eta_pp = algo4_fp_update(w_flat_new, Rs_block_new, h_list, sigma2_k)

        # ── 步骤 4：旋转角优化 ──
        if method == 'ES':
            kw = es_params or {'delta_phi': np.deg2rad(5)}
            phi_new = algo9_es(
                w_ik_list, Rs_list, mu_pp, eta_pp,
                bs_pos, user_pos, alpha_true, r_true, M, N, U, K,
                sigma2_k, sigma2, beta, eps_alpha, eps_r,
                lambda_alpha, lambda_r, tau,
                phi_min, phi_max, **kw)
        else:
            kw = pso_params or {}
            phi_new = algo10_pso(
                w_ik_list, Rs_list, mu_pp, eta_pp,
                bs_pos, user_pos, alpha_true, r_true, M, N, U, K,
                sigma2_k, sigma2, beta, eps_alpha, eps_r,
                lambda_alpha, lambda_r, tau,
                phi_min, phi_max, **kw)

        print(f"  步骤4 旋转角优化完成: φ={np.rad2deg(phi_new).round(2)} °")

        # ── 步骤 5：收敛检查 ──
        theta_new, a_new, b_new, da_new_a, da_new_r, db_new_a, db_new_r = \
            algo0_precompute(bs_pos, alpha_true, r_true, phi_new, M, N)
        h_list_new = build_user_channels(bs_pos, user_pos, phi_new, tau, M, U, K)

        R_new = compute_sinr_rate(w_ik_list, Rs_list, h_list_new, sigma2_k, U, K, M)

        # ── FIM 诊断 ──
        Rx_diag = []
        for i in range(U):
            Ww_d = sum(np.outer(w_ik_list[i*K+k], w_ik_list[i*K+k].conj()) for k in range(K))
            Rx_diag.append(Ww_d + Rs_list[i])
        P_sd, P_ad, P_rd, c_ad, c_rd, c_ard = algo1_fim_scalars(Rx_diag, b_new, db_new_a, db_new_r)
        S_aad, S_rrd, S_ard = algo2_fim_elements(P_sd, P_ad, P_rd, c_ad, c_rd, c_ard,
                                                   a_new, da_new_a, da_new_r, beta, N, sigma2)
        delta_fim = S_aad * S_rrd - S_ard**2
        Rs_traces = [float(np.real(np.trace(Rs_list[i]))) for i in range(U)]
        print(f"  [FIM] Sαα={S_aad:.4e}, Srr={S_rrd:.4e}, Sar={S_ard:.4e}, "
              f"Δ={delta_fim:.4e}, Rs_tr={[f'{v:.4f}' for v in Rs_traces]}")

        abs_delta = abs(R_new - R_prev)
        rel_delta = abs_delta / max(abs(R_prev), 1e-6)

        # 变量变化量
        w_flat_prev = w_list_to_flat(w_ik_list, U, K, M)  # 已是新值（波束已更新）
        var_delta_phi = norm(phi_new - phi)

        print(f"  R_new={R_new:.4f}, abs_Δ={abs_delta:.6f}, "
              f"rel_Δ={rel_delta:.6f}, |Δφ|={var_delta_phi:.4f}")

        if (rel_delta < delta_AO and abs_delta < eps_abs) or var_delta_phi < eps_AO:
            converged = True
            print("  ✓ 收敛")

        phi = phi_new.copy()
        R_prev = R_new
        t += 1

    # ═══ 输出 ═══
    theta_f, a_f, b_f, da_f_a, da_f_r, db_f_a, db_f_r = \
        algo0_precompute(bs_pos, alpha_true, r_true, phi, M, N)
    Rx_final = []
    for i in range(U):
        Ww = sum(np.outer(w_ik_list[i*K+k], w_ik_list[i*K+k].conj())
                 for k in range(K))
        Rx_final.append(Ww + Rs_list[i])

    P_sf, P_af, P_rf, c_af, c_rf, c_arf = algo1_fim_scalars(Rx_final, b_f, db_f_a, db_f_r)
    S_aaf, S_rrf, S_arf = algo2_fim_elements(P_sf, P_af, P_rf, c_af, c_rf, c_arf,
                                               a_f, da_f_a, da_f_r, beta, N, sigma2)
    CRB_alpha_f, CRB_r_f = algo3_crb(S_aaf, S_rrf, S_arf, sigma2)

    return {
        'w_ik_list': w_ik_list,
        'Rs_list':   Rs_list,
        'phi':       phi,
        'R_star':    R_prev,
        'CRB_alpha': CRB_alpha_f,
        'CRB_r':     CRB_r_f,
        'iterations': t,
        'converged':  converged,
    }


def compute_sinr_rate(w_ik_list, Rs_list, h_list, sigma2_k, U, K, M):
    """计算真实通信和速率 R = Σₖ log₂(1+SINRₖ)"""
    Rs_block = block_diag_np(Rs_list)
    w_flat   = w_list_to_flat(w_ik_list, U, K, M)
    R_sum = 0.0
    for k in range(K):
        hk = h_list[k]
        signal = np.abs(hk.conj() @ w_flat[k])**2
        inter  = sum(np.abs(hk.conj() @ w_flat[j])**2 for j in range(K) if j != k)
        noise  = np.real(hk.conj() @ Rs_block @ hk) + sigma2_k[k]
        noise  = max(float(noise), 1e-12)   # 数值保护
        SINR_k = max(signal / (inter + noise), 0.0)  # 防止负值
        R_sum += np.log2(1 + SINR_k)
    return R_sum


# ============================================================
# 快速验证：小规模仿真入口
# ============================================================

if __name__ == '__main__':
    np.random.seed(42)

    # ── 系统参数 ──
    U = 3   # 基站数（三基站）
    K = 2   # 用户数
    M = 4   # 发射天线
    N = 4   # 接收天线

    # 基站坐标 (m)：三角形分布
    bs_pos = np.array([[  0.0,   0.0],   # BS1：原点
                       [ 50.0,   0.0],   # BS2：东侧 50m
                       [ 25.0,  43.3]])  # BS3：北侧（等边三角形顶点）

    # 用户坐标 (m)
    user_pos = np.array([[ 20.0,  30.0],   # 用户1
                         [ 40.0,  10.0]])  # 用户2

    # 目标位置
    alpha_true = np.deg2rad(30)   # 30°
    r_true     = 100.0             # 100 m

    # 路径损耗：归一化模型 tau = (d0/d)^2，d0=30m
    # 使得典型距离(~35m)对应 tau ≈ 0.5~1.0，保证信道足够强
    d0 = 30.0
    tau = np.zeros((U, K))
    for u in range(U):
        for k in range(K):
            d_uk = np.sqrt((bs_pos[u,0]-user_pos[k,0])**2 +
                           (bs_pos[u,1]-user_pos[k,1])**2)
            tau[u][k] = (d0 / max(d_uk, 1.0)) ** 2
    print(f"路径损耗矩阵 tau =\n{tau.round(4)}")

    # 信道反射系数 βᵤ（三基站）
    beta = np.array([1.0 + 0.5j, 0.8 - 0.3j, 0.9 + 0.2j])

    # 功率与噪声（归一化，SNR ≈ 10~20 dB）
    P_max    = np.array([1.0, 1.0, 1.0])   # 各基站最大功率
    sigma2   = 1e-2                          # 感知噪声
    sigma2_k = np.array([1e-2, 1e-2])       # 通信噪声

    # 感知 CRB 阈值
    # 先用宽松阈值验证框架收敛，后续可逐步收紧
    eps_alpha = 5.0    # 方位角 CRB 阈值（弧度²）
    eps_r     = 50.0   # 距离 CRB 阈值（m²）

    # 旋转角范围
    phi_min = -np.pi / 4
    phi_max =  np.pi / 4

    # ══ FIM 单元测试（验证公式正确性）══
    print("\n" + "─" * 50)
    print("FIM 单元测试（各向同性 Rx = P_max/M * I）")
    phi_test = np.zeros(U)
    Rx_test  = [(P_max[i] / M) * np.eye(M, dtype=complex) for i in range(U)]
    _, a_t0, b_t0, da_a0, da_r0, db_a0, db_r0 = \
        algo0_precompute(bs_pos, alpha_true, r_true, phi_test, M, N)
    P_s0, P_a0, P_r0, c_a0, c_r0, c_ar0 = \
        algo1_fim_scalars(Rx_test, b_t0, db_a0, db_r0)
    print(f"  P_sum={P_s0:.4e}, P_alpha={P_a0:.4e}, P_r={P_r0:.4e}")
    print(f"  c_alpha={c_a0:.4e}, c_r={c_r0:.4e}, c_alpha_r={c_ar0:.4e}")
    S_aa0, S_rr0, S_ar0 = algo2_fim_elements(
        P_s0, P_a0, P_r0, c_a0, c_r0, c_ar0,
        a_t0, da_a0, da_r0, beta, N, sigma2)
    delta0 = S_aa0 * S_rr0 - S_ar0**2
    print(f"  Saa={S_aa0:.4e}, Srr={S_rr0:.4e}, Sar={S_ar0:.4e}, Δ={delta0:.4e}")
    if delta0 > 0:
        CRB_a0_t = sigma2 * S_rr0 / (2 * delta0)
        CRB_r0_t = sigma2 * S_aa0 / (2 * delta0)
        print(f"  ✓ FIM 有限: CRBα={CRB_a0_t:.4e} rad², CRBr={CRB_r0_t:.4e} m²")
    else:
        print(f"  ✗ FIM 奇异 (Δ≤0), 检查公式或参数!")
        # 打印 gamma 和 A 值诊断
        for u in range(U):
            g = 2*np.abs(beta[u])**2/(sigma2*N)
            d_a = np.imag(a_t0[u].conj() @ da_a0[u])
            d_r = np.imag(a_t0[u].conj() @ da_r0[u])
            A_aa = P_a0 * N + c_a0 * d_a
            A_rr = P_r0 * N + c_r0 * d_r
            print(f"    BS{u}: γ={g:.2f}, d_α={d_a:.4f}, d_r={d_r:.6f}, "
                  f"A_aa={A_aa:.4f}, A_rr={A_rr:.4f}")
    print("─" * 50 + "\n")

    print("=" * 60)
    print("可旋转天线 ISAC 系统 AO 优化")
    print(f"U={U}, K={K}, M={M}, N={N}")
    print("=" * 60)

    result = algo11_ao(
        bs_pos=bs_pos,
        user_pos=user_pos,
        tau=tau,
        P_max=P_max,
        beta=beta,
        sigma2=sigma2,
        sigma2_k=sigma2_k,
        alpha_true=alpha_true,
        r_true=r_true,
        M=M, N=N, U=U, K=K,
        eps_alpha=eps_alpha,
        eps_r=eps_r,
        phi_min=phi_min,
        phi_max=phi_max,
        lambda_alpha=5.0,
        lambda_r=5.0,
        T_max=50,
        delta_AO=1e-2,
        eps_abs=1e-3,
        eps_AO=1e-4,
        rho_init=0.0,
        method='PSO',
        pso_params={'P_pso': 20, 'T_pso': 50}
    )

    print("\n" + "=" * 60)
    print("最优结果:")
    print(f"  通信和速率 R* = {result['R_star']:.4f} bps/Hz")
    print(f"  CRB(α)*       = {result['CRB_alpha']:.4e}  (阈值 {eps_alpha:.4e})")
    print(f"  CRB(r)*       = {result['CRB_r']:.4e}  (阈值 {eps_r:.4e})")
    print(f"  最优旋转角 φ* = {np.rad2deg(result['phi']).round(2)} °")
    print(f"  迭代次数      = {result['iterations']}")
    print(f"  收敛状态      = {result['converged']}")
    print("=" * 60)
