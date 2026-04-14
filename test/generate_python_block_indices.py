"""
Generate block indices from Python MBB (seed=12345) for Julia cross-validation.

Replays the exact np.random.randint calls that make_proxy_svar_mbb would make,
saving the block indices so Julia can replay the same resampling and compare
CIs to machine precision.

Also saves the per-rep bootstrap IRFs (irf_norm) for direct comparison.
"""

import sys
sys.path.insert(0, "/workspace/Replication_mps/PythonSVAR")

import numpy as np
import os

try:
    from numba import njit
except ImportError:
    def njit(f): return f

from importlib.machinery import SourceFileLoader
proxy_mod = SourceFileLoader("proxy_svar_module",
    "/workspace/Replication_mps/PythonSVAR/31608103_proxy_svar_module.py").load_module()

# ============================================================================
# Load data (same as generate_python_bootstrap_testdata.py)
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
nBoot = 500

# ============================================================================
# Estimate proxy-SVAR
# ============================================================================
yy = Y[pp:, :]
T_eff = yy.shape[0]

xx = np.ones((T_eff, KK * pp + 1))
for lag in range(pp):
    xx[:, 1 + lag*KK : 1 + (lag+1)*KK] = Y[pp-1-lag:TT-1-lag, :]

mm = proxy_full[pp:].reshape(-1, 1)

a_hat, u_hat, covUU, covUM, h1_hat = proxy_mod.estimate_proxy_svar(yy, xx, mm)

# ============================================================================
# Generate block indices with seed=12345 (matching the exact RNG sequence)
# ============================================================================
numblocks = T_eff - blocksize + 1  # 195
numResample = int(np.ceil(T_eff / blocksize))  # 50

np.random.seed(12345)

# The MBB loop draws numResample indices per rep
block_indices = np.zeros((nBoot, numResample), dtype=int)
for boot in range(nBoot):
    for ii in range(numResample):
        block_indices[boot, ii] = np.random.randint(numblocks)

# ============================================================================
# Now replay the MBB manually with these indices to get per-rep IRFs
# ============================================================================
# Build blocks and centering (same as make_proxy_svar_mbb)
u_blocks = np.zeros((blocksize, KK, numblocks))
m_blocks = np.zeros((blocksize, 1, numblocks))
for ii in range(numblocks):
    u_blocks[:, :, ii] = u_hat[ii:blocksize+ii, :]
    m_blocks[:, :, ii] = mm[ii:blocksize+ii, :]

# Centering (only residuals, NOT proxy — this is what Python does)
u_center_block = np.zeros((blocksize, KK))
for ii in range(blocksize):
    for nn in range(KK):
        u_center_block[ii, nn] = np.mean(u_hat[ii:T_eff-blocksize+ii+1, nn])

u_center = np.zeros((numResample * blocksize, KK))
for ii in range(numResample):
    u_center[ii*blocksize:(ii+1)*blocksize, :] = u_center_block

# Initial conditions
y0 = np.zeros(KK * pp + 1)
y0[0] = 1
for lag in range(pp):
    y0[1 + lag*KK : 1 + (lag+1)*KK] = Y[pp - 1 - lag, :]

# Transpose A for simulation
AA = a_hat.T  # (1+K*p, K)

# Storage
irf_norm_store = np.zeros((KK, n_imp, nBoot))

for boot in range(nBoot):
    # Resample blocks using saved indices
    u_temp = np.empty((numResample * blocksize, KK))
    m_temp = np.empty((numResample * blocksize, 1))
    for ii in range(numResample):
        idx = block_indices[boot, ii]
        u_temp[ii*blocksize:(ii+1)*blocksize, :] = u_blocks[:, :, idx]
        m_temp[ii*blocksize:(ii+1)*blocksize, :] = m_blocks[:, :, idx]

    # Center residuals only (matching Python)
    u_temp[:, :] = u_temp - u_center

    # Truncate
    u_star = u_temp[0:T_eff, :]
    m_star = m_temp[0:T_eff, :]

    # Compute bootstrap dynamics
    irf_boot, irf_norm_boot, fevd_boot, svma_boot = \
        proxy_mod.make_boot_dynamics(AA, u_star, m_star, y0.flatten(),
                                     pp, s, n_imp)

    irf_norm_store[:, :, boot] = irf_norm_boot

# ============================================================================
# Compute percentile CIs from replay (should match original)
# ============================================================================
irf_norm_sort = np.sort(irf_norm_store, axis=2)

num16 = int(round(0.16 * nBoot))
num84 = int(round(0.84 * nBoot))
num025 = int(round(0.025 * nBoot))
num975 = int(round(0.975 * nBoot))

ci68_replay = np.empty((2, n_imp, KK))
ci95_replay = np.empty((2, n_imp, KK))
for kk in range(KK):
    ci68_replay[0, :, kk] = irf_norm_sort[kk, :, num16]
    ci68_replay[1, :, kk] = irf_norm_sort[kk, :, num84]
    ci95_replay[0, :, kk] = irf_norm_sort[kk, :, num025]
    ci95_replay[1, :, kk] = irf_norm_sort[kk, :, num975]

# Verify replay matches original
ci68_orig = np.loadtxt(os.path.join(outdir, "jl_crossval_mbb_ci68_irf_norm.csv"), delimiter=",")
ci95_orig = np.loadtxt(os.path.join(outdir, "jl_crossval_mbb_ci95_irf_norm.csv"), delimiter=",")

ci68_replay_flat = ci68_replay.reshape(-1, KK)
ci95_replay_flat = ci95_replay.reshape(-1, KK)

print(f"Replay vs original max diff (68%): {np.max(np.abs(ci68_replay_flat - ci68_orig)):.2e}")
print(f"Replay vs original max diff (95%): {np.max(np.abs(ci95_replay_flat - ci95_orig)):.2e}")

# ============================================================================
# Save block indices and per-rep irf_norm draws
# ============================================================================
# Block indices: (nBoot, numResample) — 0-indexed as Python uses them
np.savetxt(os.path.join(outdir, "jl_crossval_mbb_block_indices.csv"),
    block_indices, delimiter=",", fmt="%d", comments="")

# Per-rep irf_norm: save as (nBoot, KK * n_imp) for easy loading
# Layout: [boot, k*n_imp + h] = irf_norm_store[k, h, boot]
irf_norm_flat = np.zeros((nBoot, KK * n_imp))
for boot in range(nBoot):
    for k in range(KK):
        for h in range(n_imp):
            irf_norm_flat[boot, k * n_imp + h] = irf_norm_store[k, h, boot]

np.savetxt(os.path.join(outdir, "jl_crossval_mbb_irf_norm_draws.csv"),
    irf_norm_flat, delimiter=",", comments="")

print(f"\nSaved to {outdir}/:")
print(f"  block_indices: ({nBoot}, {numResample}) = {block_indices.shape}")
print(f"  irf_norm_draws: ({nBoot}, {KK * n_imp}) = {irf_norm_flat.shape}")
print(f"  Block index range: [{block_indices.min()}, {block_indices.max()}]")
print(f"  numblocks = {numblocks}, blocksize = {blocksize}")
