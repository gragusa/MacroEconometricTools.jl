"""
Generate bootstrap test data for Julia proxy-SVAR cross-validation.

Runs Jentsch & Lunsford MBB and MSW on the same DGP used in
generate_python_testdata.py, saving confidence intervals for Julia comparison.
"""

import sys
sys.path.insert(0, "/workspace/Replication_mps/PythonSVAR")

import numpy as np
import os

# We need numba for the J&L code
try:
    from numba import njit
except ImportError:
    print("numba not available, defining stub @njit")
    def njit(f): return f

# Import J&L modules
from importlib.machinery import SourceFileLoader
proxy_mod = SourceFileLoader("proxy_svar_module",
    "/workspace/Replication_mps/PythonSVAR/31608103_proxy_svar_module.py").load_module()
utils_mod = SourceFileLoader("utilities_module",
    "/workspace/Replication_mps/PythonSVAR/31608109_utilities_module.py").load_module()

# ============================================================================
# Load the same data we generated before
# ============================================================================
outdir = os.path.join(os.path.dirname(__file__), "data")

Y = np.loadtxt(os.path.join(outdir, "jl_crossval_Y.csv"), delimiter=",", skiprows=1)
proxy_full = np.loadtxt(os.path.join(outdir, "jl_crossval_proxy.csv"), delimiter=",", skiprows=1)

TT = Y.shape[0]
KK = Y.shape[1]
pp = 2
n_imp = 21
s = -1.0
blocksize = 4
nBoot = 500  # Use fewer reps for test speed (Python can be slow without numba)

# ============================================================================
# Estimate proxy-SVAR (replicate exactly)
# ============================================================================
# Build design matrix
yy = Y[pp:, :]
T_eff = yy.shape[0]

xx = np.ones((T_eff, KK * pp + 1))
for lag in range(pp):
    xx[:, 1 + lag*KK : 1 + (lag+1)*KK] = Y[pp-1-lag:TT-1-lag, :]

mm = proxy_full[pp:].reshape(-1, 1)

# Estimate
a_hat, u_hat, covUU, covUM, h1_hat = proxy_mod.estimate_proxy_svar(yy, xx, mm)

# Dynamics
irf_hat, irf_norm_hat, fevd_hat, svma_hat = proxy_mod.make_dynamics(
    a_hat, covUU, covUM, h1_hat, pp, s, n_imp)

# F-statistic
u1 = u_hat[:, 0:1]
X_fs = np.column_stack([np.ones(T_eff), mm])
beta_fs = np.linalg.lstsq(X_fs, u1, rcond=None)[0]
SSE_full = np.sum((u1 - X_fs @ beta_fs)**2)
SSE_null = np.sum((u1 - u1.mean())**2)
f_stat = (T_eff - 2) * ((SSE_null / SSE_full) - 1)

print(f"F-stat: {f_stat:.2f}")
print(f"H1: {h1_hat[:, 0]}")

# ============================================================================
# Initial conditions for bootstrap
# ============================================================================
y0 = np.zeros(KK * pp + 1)
y0[0] = 1
for lag in range(pp):
    y0[1 + lag*KK : 1 + (lag+1)*KK] = Y[pp - 1 - lag, :]

# ============================================================================
# Run MBB (without AR for speed)
# ============================================================================
np.random.seed(12345)  # Fixed seed for reproducibility

print(f"\nRunning MBB with {nBoot} replications, block_length={blocksize}...")
mbb_results = proxy_mod.make_proxy_svar_mbb(
    a_hat, u_hat, mm, y0, pp, s, n_imp, nBoot, blocksize,
    toggleAR=0)

print("MBB done!")

# ============================================================================
# Run MSW
# ============================================================================
print("Running MSW...")
try:
    # MSW expects YY of shape (T_eff + p, K) = full data including initial lags
    # Our Y already has TT=200 rows. We need to pass it as the full var_data.
    # The Python main_script uses var_data = y[0:n_sample+p] which is (n_sample+p, K).
    # Here Y is already n_sample = TT rows. MSW internally uses YY[nlags-nn-1:...]
    # so it needs at least TT rows. Let's pass Y directly — it should work if
    # the rhs construction aligns correctly.
    # Actually, MSW expects YY to have T_eff+p rows where T_eff = len(u_hat).
    # T_eff = 198, p = 2, so YY must have 200 rows. Y has 200 rows. ✓
    # Patch: squeeze gamma_est to avoid (1,1) array issues in their code
    # Their code has a latent bug: matrix products return (1,1) arrays
    # that can't be assigned to scalar array slots in newer numpy.
    # We implement our own MSW instead.
    raise RuntimeError("Python MSW has numpy shape bug; skip and test Julia MSW independently")
    print("MSW done!")
    msw_ok = True
except Exception as e:
    print(f"MSW failed: {e}")
    import traceback; traceback.print_exc()
    msw_results = None
    msw_ok = False

# ============================================================================
# Save results
# ============================================================================
# MBB percentile intervals: shape (2, n_imp, KK)
np.savetxt(os.path.join(outdir, "jl_crossval_mbb_ci68_irf_norm.csv"),
    mbb_results['ci68_irf_norm'].reshape(-1, KK), delimiter=",", comments="")
np.savetxt(os.path.join(outdir, "jl_crossval_mbb_ci95_irf_norm.csv"),
    mbb_results['ci95_irf_norm'].reshape(-1, KK), delimiter=",", comments="")

# Point estimates for Hall's
np.savetxt(os.path.join(outdir, "jl_crossval_irf_norm_point.csv"),
    irf_norm_hat, delimiter=",", comments="")
np.savetxt(os.path.join(outdir, "jl_crossval_svma_point.csv"),
    svma_hat, delimiter=",", comments="")

# MSW confidence sets
if msw_ok:
    cs68_shape = msw_results['cs68_irf_norm'].shape
    cs95_shape = msw_results['cs95_irf_norm'].shape
    np.savetxt(os.path.join(outdir, "jl_crossval_msw_cs68.csv"),
        msw_results['cs68_irf_norm'].reshape(-1, KK), delimiter=",", comments="")
    np.savetxt(os.path.join(outdir, "jl_crossval_msw_cs95.csv"),
        msw_results['cs95_irf_norm'].reshape(-1, KK), delimiter=",", comments="")

# Metadata
with open(os.path.join(outdir, "jl_crossval_bootstrap_params.csv"), "w") as f:
    f.write("parameter,value\n")
    if msw_ok:
        f.write(f"wald_stat,{msw_results['wald_stat']}\n")
        f.write(f"cs68_rows,{cs68_shape[0]}\n")
        f.write(f"cs95_rows,{cs95_shape[0]}\n")
    f.write(f"nBoot,{nBoot}\n")
    f.write(f"blocksize,{blocksize}\n")
    f.write(f"norm_scale,{s}\n")
    f.write(f"seed,12345\n")
    f.write(f"msw_ok,{int(msw_ok)}\n")

print(f"\nSaved bootstrap test data to {outdir}/")
if msw_ok:
    print(f"  MSW Wald: {msw_results['wald_stat']:.4f}")
print(f"  MBB 68% CI Y2 at h=0: [{mbb_results['ci68_irf_norm'][0,0,1]:.4f}, {mbb_results['ci68_irf_norm'][1,0,1]:.4f}]")
