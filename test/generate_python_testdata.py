"""
Generate test data for Julia proxy-SVAR cross-validation.

Uses the exact DGP and estimation code from Jentsch & Lunsford (2022)
"Asymptotically Valid Bootstrap Inference for Proxy SVARs".

The generated data is saved as CSV files in test/data/ for Julia to load.
"""

import numpy as np
import os

# ============================================================================
# DGP Parameters (exact Jentsch & Lunsford specification)
# ============================================================================
np.random.seed(0)  # for reproducibility

KK = 2   # number of VAR variables
pp = 2   # number of lags
TT = 200 # sample size (their main simulation)
burn = 1000

# VAR coefficients: A = [constant | A1 | A2]
# Python format: A is KK x (KK*pp + 1), row = equation
A_true = np.array([
    [0.0,  0.44,  0.66, -0.18,  0.0],
    [0.0, -0.11,  1.32, -0.18, -0.09]
])

# Structural impact matrix H (lower triangular-ish)
H_true = np.array([
    [np.sqrt(2)/2,                    np.sqrt(2)/2],
    [(np.sqrt(2)-np.sqrt(6))/4, (np.sqrt(2)+np.sqrt(6))/4]
])

# Stochastic volatility parameters
rho = 0.85
sigma_e = 0.15
log_sig_bar = -sigma_e**2 / (1 - rho**2)

# Proxy strength (DGP1 = strong)
psi = 1.0

# ============================================================================
# Data Generation (reproduce make_data_sv exactly)
# ============================================================================
TT_total = TT + burn

# Innovations to volatility
ee = sigma_e * np.random.standard_normal(size=(TT_total, KK))

# Log volatility AR(1)
mu_big = log_sig_bar * np.ones(KK)
log_sig = np.zeros((TT_total, KK))
log_sig[0, :] = (1 - rho) * mu_big + ee[0, :]
for ii in range(TT_total - 1):
    log_sig[ii+1, :] = (1 - rho) * mu_big + rho * log_sig[ii, :] + ee[ii+1, :]

# Structural shocks with stochastic volatility
ww = np.random.standard_normal(size=(TT_total, KK))
eps = ww * np.exp(log_sig)

# VAR innovations
uu = eps @ H_true.T

# Generate VAR variables
YY = np.empty((TT_total, KK))
history = np.zeros((1, KK * pp + 1))
history[0, 0] = 1  # constant

for ii in range(TT_total):
    YY[ii, :] = history @ A_true.T + uu[ii, :]
    history[0, KK+1:KK*pp+1] = history[0, 1:KK*(pp-1)+1]
    history[0, 1:KK+1] = YY[ii, :]

# Proxy noise
noise = np.random.standard_normal(size=(TT_total, 1))

# Drop burn-in
YY = YY[burn:, :]
eps = eps[burn:, :]
noise = noise[burn:, :]

# Construct proxy variable
proxy = psi * eps[:, 0:1] + noise

# ============================================================================
# Estimation (reproduce estimate_proxy_svar exactly)
# ============================================================================
# Construct design matrix X = [1, y_{t-1}, ..., y_{t-p}]
yy = YY[pp:, :]  # dependent variable (T-p x K)
T_eff = yy.shape[0]

# Build lagged matrix
xx = np.ones((T_eff, KK * pp + 1))
for lag in range(pp):
    xx[:, 1 + lag*KK : 1 + (lag+1)*KK] = YY[pp-1-lag:TT-1-lag, :]

# Proxy aligned with residuals
mm = proxy[pp:, :]

# Estimate VAR by OLS
mat1 = xx.T @ xx
mat2 = xx.T @ yy
A_est = np.linalg.solve(mat1, mat2)

# VAR residuals
U_est = yy - xx @ A_est

# Covariance matrices (NO degrees-of-freedom correction, matches Python)
covUU_est = (U_est.T @ U_est) / T_eff
covUM_est = (U_est.T @ mm) / T_eff

# Proxy SVAR identification
phi_sq = covUM_est.T @ np.linalg.solve(covUU_est, covUM_est)
phi = np.sqrt(phi_sq[0, 0])
H1_est = covUM_est / phi

# Transpose A to match Python convention: A_est is KK x (KK*pp+1)
A_est_T = A_est.T

# F-statistic (first stage: regress u1 on proxy)
u1 = U_est[:, 0:1]
X_fs = np.column_stack([np.ones(T_eff), mm])
beta_fs = np.linalg.lstsq(X_fs, u1, rcond=None)[0]
u1_hat = X_fs @ beta_fs
SSE_full = np.sum((u1 - u1_hat)**2)
SSE_null = np.sum((u1 - u1.mean())**2)
F_stat = (T_eff - 2) * ((SSE_null / SSE_full) - 1) / 1

# ============================================================================
# IRF computation
# ============================================================================
n_imp = 21  # horizons 0-20

# Companion matrix
comp = np.zeros((KK * pp, KK * pp))
comp[0:KK, :] = A_est_T[:, 1:KK*pp+1]
comp[KK:KK*pp, 0:KK*(pp-1)] = np.eye(KK * (pp - 1))

# IRFs
irf = np.empty((KK, n_imp))
irf[:, 0] = H1_est[:, 0]

comp_power = np.eye(KK * pp)
for hh in range(1, n_imp):
    comp_power = comp_power @ comp
    irf[:, hh] = (comp_power[0:KK, 0:KK] @ H1_est[:, 0:1])[:, 0]

# Normalized IRF (impact = -1)
s = -1.0
irf_norm = np.empty((KK, n_imp))
for hh in range(n_imp):
    irf_norm[:, hh] = s * irf[:, hh] / irf[0, 0]

# ============================================================================
# Save everything
# ============================================================================
outdir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(outdir, exist_ok=True)

# Save data
np.savetxt(os.path.join(outdir, "jl_crossval_Y.csv"), YY, delimiter=",",
           header="Y1,Y2", comments="")
np.savetxt(os.path.join(outdir, "jl_crossval_proxy.csv"), proxy, delimiter=",",
           header="proxy", comments="")
np.savetxt(os.path.join(outdir, "jl_crossval_eps.csv"), eps, delimiter=",",
           header="eps1,eps2", comments="")

# Save estimation results
np.savetxt(os.path.join(outdir, "jl_crossval_A_est.csv"), A_est_T, delimiter=",",
           comments="")  # KK x (KK*pp+1)
np.savetxt(os.path.join(outdir, "jl_crossval_U_est.csv"), U_est, delimiter=",",
           comments="")  # T_eff x KK
np.savetxt(os.path.join(outdir, "jl_crossval_covUU.csv"), covUU_est, delimiter=",",
           comments="")
np.savetxt(os.path.join(outdir, "jl_crossval_covUM.csv"), covUM_est, delimiter=",",
           comments="")
np.savetxt(os.path.join(outdir, "jl_crossval_H1.csv"), H1_est, delimiter=",",
           comments="")
np.savetxt(os.path.join(outdir, "jl_crossval_irf.csv"), irf, delimiter=",",
           comments="")  # KK x n_imp
np.savetxt(os.path.join(outdir, "jl_crossval_irf_norm.csv"), irf_norm, delimiter=",",
           comments="")

# Save scalar parameters
with open(os.path.join(outdir, "jl_crossval_params.csv"), "w") as f:
    f.write("parameter,value\n")
    f.write(f"T,{TT}\n")
    f.write(f"K,{KK}\n")
    f.write(f"p,{pp}\n")
    f.write(f"T_eff,{T_eff}\n")
    f.write(f"psi,{psi}\n")
    f.write(f"phi,{phi}\n")
    f.write(f"F_stat,{F_stat}\n")
    f.write(f"n_imp,{n_imp}\n")
    f.write(f"norm_scale,{s}\n")
    f.write(f"H1_1,{H1_est[0,0]}\n")
    f.write(f"H1_2,{H1_est[1,0]}\n")

# Save true parameters
np.savetxt(os.path.join(outdir, "jl_crossval_A_true.csv"), A_true, delimiter=",",
           comments="")
np.savetxt(os.path.join(outdir, "jl_crossval_H_true.csv"), H_true, delimiter=",",
           comments="")

print(f"Saved test data to {outdir}/")
print(f"  T={TT}, K={KK}, p={pp}, T_eff={T_eff}")
print(f"  phi={phi:.6f}")
print(f"  F_stat={F_stat:.2f}")
print(f"  H1_est = [{H1_est[0,0]:.6f}, {H1_est[1,0]:.6f}]")
print(f"  H1_true = [{H_true[0,0]:.6f}, {H_true[1,0]:.6f}]")
print(f"  H1_ratio (est) = {H1_est[1,0]/H1_est[0,0]:.6f}")
print(f"  H1_ratio (true) = {H_true[1,0]/H_true[0,0]:.6f}")
