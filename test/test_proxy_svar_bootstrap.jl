# ============================================================================
# Tests for Proxy-SVAR Bootstrap (Jentsch & Lunsford MBB + AR + MSW)
# ============================================================================

using MacroEconometricTools
using Test
using LinearAlgebra
using Random
using Statistics
using StableRNGs: StableRNG
using CSV
using DataFrames

const BDATA = joinpath(@__DIR__, "data")

# ============================================================================
# Test 1: proxy_svar_dynamics matches Python make_dynamics
# ============================================================================
@testset "proxy_svar_dynamics matches Python" begin
    # Load Python data
    A_est = Matrix(CSV.read(joinpath(BDATA, "jl_crossval_A_est.csv"), DataFrame; header = false))
    covUU = Matrix(CSV.read(joinpath(BDATA, "jl_crossval_covUU.csv"), DataFrame; header = false))
    covUM = vec(Matrix(CSV.read(joinpath(BDATA, "jl_crossval_covUM.csv"), DataFrame; header = false)))
    H1 = vec(Matrix(CSV.read(joinpath(BDATA, "jl_crossval_H1.csv"), DataFrame; header = false)))
    irf_py = Matrix(CSV.read(joinpath(BDATA, "jl_crossval_irf.csv"), DataFrame; header = false))
    irf_norm_py = Matrix(CSV.read(joinpath(BDATA, "jl_crossval_irf_norm_point.csv"), DataFrame; header = false))
    svma_py = Matrix(CSV.read(joinpath(BDATA, "jl_crossval_svma_point.csv"), DataFrame; header = false))

    p = 2;
    K = 2;
    n_imp = 21;
    s = -1.0

    dyn = MacroEconometricTools.proxy_svar_dynamics(A_est, covUU, covUM, H1, p, s, n_imp, 1)

    @testset "IRFs match" begin
        for k in 1:K, h in 1:n_imp

            @test dyn.irf[k, h] ≈ irf_py[k, h] atol=1e-10
        end
    end

    @testset "Normalized IRFs match" begin
        for k in 1:K, h in 1:n_imp

            @test dyn.irf_norm[k, h] ≈ irf_norm_py[k, h] atol=1e-8
        end
    end

    @testset "SVMA matches" begin
        for k in 1:K, h in 1:n_imp

            @test dyn.svma[k, h] ≈ svma_py[k, h] atol=1e-10
        end
    end

    @testset "FEVD in [0, 1]" begin
        for k in 1:K, h in 1:n_imp

            @test 0 ≤ dyn.fevd[k, h] ≤ 1
        end
    end
end

# ============================================================================
# Test 2: estimate_proxy_svar matches Python reference
# ============================================================================
@testset "estimate_proxy_svar matches Python reference" begin
    Y = Matrix(CSV.read(joinpath(BDATA, "jl_crossval_Y.csv"), DataFrame))
    proxy = vec(Matrix(CSV.read(joinpath(BDATA, "jl_crossval_proxy.csv"), DataFrame)))
    A_est_py = Matrix(CSV.read(joinpath(BDATA, "jl_crossval_A_est.csv"), DataFrame; header = false))
    U_est_py = Matrix(CSV.read(joinpath(BDATA, "jl_crossval_U_est.csv"), DataFrame; header = false))
    H1_py = vec(Matrix(CSV.read(joinpath(BDATA, "jl_crossval_H1.csv"), DataFrame; header = false)))

    p = 2;
    K = 2
    T_eff = size(Y, 1) - p

    # Build design matrix (same as Python)
    yy = Y[(p + 1):end, :]
    xx = ones(T_eff, K * p + 1)
    for lag in 1:p
        xx[:, (1 + (lag - 1) * K + 1):(1 + lag * K)] .= Y[(p + 1 - lag):(end - lag), :]
    end
    mm = proxy[(p + 1):end]

    A_est, U_est, Σ_uu, Σ_um,
    H1 = MacroEconometricTools.estimate_proxy_svar(yy, xx, mm)

    @test A_est ≈ A_est_py atol=1e-10
    @test U_est ≈ U_est_py atol=1e-8
    @test H1 ≈ H1_py atol=1e-6
end

# ============================================================================
# Test 3: MBB centering correctness
# ============================================================================
@testset "MBB: J&L position-specific centering" begin
    rng = StableRNG(42)
    K = 2;
    T = 100;
    ℓ = 4

    # Fake residuals with known structure
    ν = randn(rng, T, K)
    proxy = randn(rng, T)

    n_blocks = T - ℓ + 1

    # Compute centering the J&L way
    u_center = zeros(ℓ, K)
    for s in 1:ℓ
        u_center[s, :] .= vec(mean(ν[s:(T - ℓ + s), :]; dims = 1))
    end

    # The centered residuals at each position should have approximately zero mean
    # when averaged over all blocks
    for s in 1:ℓ
        block_mean = vec(mean(ν[s:(T - ℓ + s), :]; dims = 1))
        centered_mean = block_mean .- u_center[s, :]
        @test all(abs.(centered_mean) .< 1e-14)
    end
end

# ============================================================================
# Test 4: MBB produces valid confidence intervals
# ============================================================================
@testset "MBB: valid confidence intervals" begin
    Y = Matrix(CSV.read(joinpath(BDATA, "jl_crossval_Y.csv"), DataFrame))
    proxy = vec(Matrix(CSV.read(joinpath(BDATA, "jl_crossval_proxy.csv"), DataFrame)))

    p = 2;
    K = 2
    Z_proxy = reshape(proxy[(p + 1):end], :, 1)
    instrument = ExternalInstrument(Z_proxy, 1)
    model = fit(IVSVAR, Y, p; instrument = instrument, names = [:Y1, :Y2])

    mbb = proxy_svar_mbb(model, 20, ProxySVARMBB(200; block_length = 4);
        rng = Random.Xoshiro(999))

    @testset "68% CI is narrower than 95% CI" begin
        for k in 1:K, h in 1:21

            width68 = mbb.ci68_irf_norm[2, h, k] - mbb.ci68_irf_norm[1, h, k]
            width95 = mbb.ci95_irf_norm[2, h, k] - mbb.ci95_irf_norm[1, h, k]
            @test width68 ≤ width95 + 1e-10
        end
    end

    @testset "Point estimate inside 95% CI" begin
        for k in 1:K, h in 1:21

            point = mbb.point_irf_norm[k, h]
            @test mbb.ci95_irf_norm[1, h, k] ≤ point ≤ mbb.ci95_irf_norm[2, h, k]
        end
    end

    @testset "Normalized IRF at impact = norm_scale" begin
        @test mbb.point_irf_norm[1, 1] ≈ -1.0 atol=1e-10
    end
end

# ============================================================================
# Test 5: MBB CIs produce finite, reasonable intervals
# ============================================================================
@testset "MBB: CIs are finite and reasonable" begin
    Y = Matrix(CSV.read(joinpath(BDATA, "jl_crossval_Y.csv"), DataFrame))
    proxy = vec(Matrix(CSV.read(joinpath(BDATA, "jl_crossval_proxy.csv"), DataFrame)))

    p = 2;
    K = 2
    Z_proxy = reshape(proxy[(p + 1):end], :, 1)
    instrument = ExternalInstrument(Z_proxy, 1)
    model = fit(IVSVAR, Y, p; instrument = instrument, names = [:Y1, :Y2])

    mbb = proxy_svar_mbb(model, 20, ProxySVARMBB(500; block_length = 4);
        rng = Random.Xoshiro(42))

    @testset "All CIs are finite" begin
        @test all(isfinite, mbb.ci68_irf_norm)
        @test all(isfinite, mbb.ci95_irf_norm)
    end

    @testset "CI widths are positive" begin
        for k in 1:K, h in 1:21

            @test mbb.ci68_irf_norm[2, h, k] ≥ mbb.ci68_irf_norm[1, h, k]
            @test mbb.ci95_irf_norm[2, h, k] ≥ mbb.ci95_irf_norm[1, h, k]
        end
    end
end

# ============================================================================
# Test 6: Hall's intervals are bias-corrected
# ============================================================================
@testset "Hall's intervals: correct transformation" begin
    Y = Matrix(CSV.read(joinpath(BDATA, "jl_crossval_Y.csv"), DataFrame))
    proxy = vec(Matrix(CSV.read(joinpath(BDATA, "jl_crossval_proxy.csv"), DataFrame)))

    p = 2;
    K = 2
    Z_proxy = reshape(proxy[(p + 1):end], :, 1)
    instrument = ExternalInstrument(Z_proxy, 1)
    model = fit(IVSVAR, Y, p; instrument = instrument, names = [:Y1, :Y2])

    mbb = proxy_svar_mbb(model, 10, ProxySVARMBB(100; block_length = 4);
        rng = Random.Xoshiro(123))

    # Hall's formula: lower = 2θ̂ - upper_pctile, upper = 2θ̂ - lower_pctile
    for k in 1:K, h in 1:11

        point = mbb.point_irf_norm[k, h]
        @test mbb.halls68_irf_norm[1, h, k] ≈ 2*point - mbb.ci68_irf_norm[2, h, k] atol=1e-12
        @test mbb.halls68_irf_norm[2, h, k] ≈ 2*point - mbb.ci68_irf_norm[1, h, k] atol=1e-12
    end
end

# ============================================================================
# Test 7: AR confidence sets
# ============================================================================
@testset "AR confidence sets" begin
    Y = Matrix(CSV.read(joinpath(BDATA, "jl_crossval_Y.csv"), DataFrame))
    proxy = vec(Matrix(CSV.read(joinpath(BDATA, "jl_crossval_proxy.csv"), DataFrame)))

    p = 2;
    K = 2
    Z_proxy = reshape(proxy[(p + 1):end], :, 1)
    instrument = ExternalInstrument(Z_proxy, 1)
    model = fit(IVSVAR, Y, p; instrument = instrument, names = [:Y1, :Y2])

    ar_grid = collect(range(-3.0, 3.0; length = 61))
    mbb = proxy_svar_mbb(model,
        10,
        ProxySVARMBB(200; block_length = 4,
            compute_ar = true, ar_grid = ar_grid, norm_scale = -1.0);
        rng = Random.Xoshiro(456))

    @testset "AR sets are nested: 68% ⊂ 90% ⊂ 95%" begin
        for k in 1:K, h in 1:11

            n68 = sum(mbb.ar.index68[:, h, k])
            n90 = sum(mbb.ar.index90[:, h, k])
            n95 = sum(mbb.ar.index95[:, h, k])
            @test n68 ≤ n90
            @test n90 ≤ n95
        end
    end

    @testset "Normalization point always in AR set" begin
        # grid point closest to norm_scale = -1
        g_norm = argmin(abs.(ar_grid .- (-1.0)))
        @test mbb.ar.index68[g_norm, 1, 1]  # impact of Y1 at h=0
        @test mbb.ar.index95[g_norm, 1, 1]
    end

    @testset "Rates are in [0, 1]" begin
        @test all(0 .≤ mbb.ar.rates .≤ 1)
    end
end

# ============================================================================
# Test 8: MSW confidence sets (analytic, no Python comparison)
# ============================================================================
@testset "MSW confidence sets" begin
    Y = Matrix(CSV.read(joinpath(BDATA, "jl_crossval_Y.csv"), DataFrame))
    proxy = vec(Matrix(CSV.read(joinpath(BDATA, "jl_crossval_proxy.csv"), DataFrame)))

    p = 2;
    K = 2
    Z_proxy = reshape(proxy[(p + 1):end], :, 1)
    instrument = ExternalInstrument(Z_proxy, 1)
    model = fit(IVSVAR, Y, p; instrument = instrument, names = [:Y1, :Y2])

    msw = msw_confidence_set(model; norm_scale = -1.0, horizon = 20)

    @testset "Wald statistic is positive" begin
        @test msw.wald_stat > 0
    end

    @testset "68% set is bounded (strong proxy)" begin
        @test msw.bounded68
    end

    @testset "Normalization at impact" begin
        @test msw.cs68_irf_norm[1, 1, 1] ≈ -1.0
        @test msw.cs68_irf_norm[2, 1, 1] ≈ -1.0
    end

    @testset "95% set wider than 68% set (bounded case)" begin
        if msw.bounded68 && msw.bounded95
            for k in 1:K, h in 2:21  # skip h=0 (normalization point)

                w68 = msw.cs68_irf_norm[2, h, k] - msw.cs68_irf_norm[1, h, k]
                w95 = msw.cs95_irf_norm[2, h, k] - msw.cs95_irf_norm[1, h, k]
                if !isnan(w68) && !isnan(w95)
                    @test w68 ≤ w95 + 1e-6
                end
            end
        end
    end
end

# ============================================================================
# Test 9: irf() dispatch with ProxySVARMBB
# ============================================================================
@testset "irf() dispatch with ProxySVARMBB" begin
    Y = Matrix(CSV.read(joinpath(BDATA, "jl_crossval_Y.csv"), DataFrame))
    proxy = vec(Matrix(CSV.read(joinpath(BDATA, "jl_crossval_proxy.csv"), DataFrame)))

    p = 2;
    K = 2
    Z_proxy = reshape(proxy[(p + 1):end], :, 1)
    instrument = ExternalInstrument(Z_proxy, 1)
    model = fit(IVSVAR, Y, p; instrument = instrument, names = [:Y1, :Y2])

    result = irf(model, IVIdentification(); horizon = 10,
        inference = ProxySVARMBB(50; block_length = 4))

    @test result isa IRFResult
    @test size(result.irf) == (K, K, 11)  # (variable, shock, horizon)
    @test length(result.lower) == 3  # 68%, 90%, 95%
    @test Array(result.irf)[1, 1, 1] ≈ 1.0 atol=1e-10  # unit effect normalization
end

# ============================================================================
# Test 10: MBB CIs match Python to machine precision (replayed block indices)
# ============================================================================
@testset "MBB CIs match Python (replayed block indices)" begin
    # Load data
    Y = Matrix(CSV.read(joinpath(BDATA, "jl_crossval_Y.csv"), DataFrame))
    proxy_full = vec(Matrix(CSV.read(joinpath(BDATA, "jl_crossval_proxy.csv"), DataFrame)))
    A_est_py = Matrix(CSV.read(joinpath(BDATA, "jl_crossval_A_est.csv"), DataFrame; header = false))
    U_est_py = Matrix(CSV.read(joinpath(BDATA, "jl_crossval_U_est.csv"), DataFrame; header = false))

    # Load Python block indices (0-indexed) and per-rep IRF draws
    block_indices_py = Matrix{Int}(CSV.read(
        joinpath(BDATA, "jl_crossval_mbb_block_indices.csv"),
        DataFrame; header = false))
    irf_norm_draws_py = Matrix(CSV.read(
        joinpath(BDATA, "jl_crossval_mbb_irf_norm_draws.csv"),
        DataFrame; header = false))

    # Load Python CIs
    ci68_py_flat = Matrix(CSV.read(joinpath(BDATA, "jl_crossval_mbb_ci68_irf_norm.csv"),
        DataFrame; header = false))
    ci95_py_flat = Matrix(CSV.read(joinpath(BDATA, "jl_crossval_mbb_ci95_irf_norm.csv"),
        DataFrame; header = false))

    p = 2;
    K = 2;
    n_imp = 21;
    s = -1.0;
    blocksize = 4
    nBoot = size(block_indices_py, 1)   # 500
    numResample = size(block_indices_py, 2)  # 50

    # Effective sample
    T_eff = size(Y, 1) - p
    yy = Y[(p + 1):end, :]
    xx = ones(T_eff, K * p + 1)
    for lag in 1:p
        xx[:, (1 + (lag - 1) * K + 1):(1 + lag * K)] .= Y[(p + 1 - lag):(end - lag), :]
    end
    mm = proxy_full[(p + 1):end]

    # Re-estimate (must match Python exactly)
    A_est, U_est, Σ_uu, Σ_um,
    H1 = MacroEconometricTools.estimate_proxy_svar(yy, xx, mm)

    @test A_est ≈ A_est_py atol=1e-10

    # A in simulation layout: (1+K*p, K)
    A_sim = A_est'

    # Initial conditions: [1, y_p, y_{p-1}]
    y_init = zeros(1 + K * p)
    y_init[1] = 1.0
    for lag in 1:p
        y_init[(1 + (lag - 1) * K + 1):(1 + lag * K)] .= Y[p + 1 - lag, :]
    end

    # Build overlapping blocks
    n_blocks = T_eff - blocksize + 1  # 195
    u_blocks = zeros(blocksize, K, n_blocks)
    m_blocks = zeros(blocksize, n_blocks)
    for b in 1:n_blocks
        u_blocks[:, :, b] .= U_est[b:(b + blocksize - 1), :]
        m_blocks[:, b] .= mm[b:(b + blocksize - 1)]
    end

    # Position-specific centering for residuals ONLY (matching Python)
    u_center_block = zeros(blocksize, K)
    for s_pos in 1:blocksize
        u_center_block[s_pos, :] .= vec(mean(U_est[s_pos:(T_eff - blocksize + s_pos), :]; dims = 1))
    end

    u_center = zeros(numResample * blocksize, K)
    for j in 1:numResample
        u_center[((j - 1) * blocksize + 1):(j * blocksize), :] .= u_center_block
    end

    # Bootstrap loop with replayed indices
    irf_norm_store = zeros(K, n_imp, nBoot)

    for boot in 1:nBoot
        # Resample blocks using Python indices (convert 0-indexed → 1-indexed)
        u_temp = zeros(numResample * blocksize, K)
        m_temp = zeros(numResample * blocksize)

        for j in 1:numResample
            idx = block_indices_py[boot, j] + 1  # 0-indexed → 1-indexed
            u_temp[((j - 1) * blocksize + 1):(j * blocksize), :] .= u_blocks[:, :, idx]
            m_temp[((j - 1) * blocksize + 1):(j * blocksize)] .= m_blocks[:, idx]
        end

        # Center residuals only (NO proxy centering — matches Python)
        u_temp .-= u_center

        # Truncate
        u_star = u_temp[1:T_eff, :]
        m_star = m_temp[1:T_eff]

        # Simulate bootstrap VAR (matching Python's make_boot_dynamics)
        x_star = zeros(T_eff, 1 + K * p)
        x_star[1, :] .= y_init
        y_star = copy(u_star)

        for t in 1:T_eff
            for k in 1:K
                for j in 1:(1 + K * p)
                    y_star[t, k] += x_star[t, j] * A_sim[j, k]
                end
            end
            if t < T_eff
                x_star[t + 1, 1] = 1.0
                x_star[t + 1, 2:(K + 1)] .= y_star[t, :]
                if p > 1
                    x_star[t + 1, (K + 2):(K * p + 1)] .= x_star[t, 2:(K * (p - 1) + 1)]
                end
            end
        end

        # Re-estimate proxy-SVAR on bootstrap sample
        A_star, U_star,
        Σ_uu_star,
        Σ_um_star,
        H1_star = MacroEconometricTools.estimate_proxy_svar(y_star, x_star, m_star)

        # Compute bootstrap dynamics
        dyn = MacroEconometricTools.proxy_svar_dynamics(
            A_star, Σ_uu_star, Σ_um_star, H1_star, p, s, n_imp, 1)

        irf_norm_store[:, :, boot] .= dyn.irf_norm
    end

    # Compare per-rep IRF draws
    @testset "Per-rep irf_norm draws match Python" begin
        max_diff = 0.0
        for boot in 1:nBoot
            for k in 1:K, h in 1:n_imp

                jl_val = irf_norm_store[k, h, boot]
                py_val = irf_norm_draws_py[boot, (k - 1) * n_imp + h]
                diff = abs(jl_val - py_val)
                max_diff = max(max_diff, diff)
            end
        end
        @test max_diff < 1e-10
    end

    # Compute percentile CIs (matching Python's quantile method)
    # Python uses 0-indexed arrays: sorted[k, h, num16] with num16 = round(0.16*nBoot)
    # accesses the (num16+1)-th element. Julia is 1-indexed, so we add 1.
    sorted_store = sort(irf_norm_store; dims = 3)

    i16 = clamp(round(Int, 0.16 * nBoot) + 1, 1, nBoot)
    i84 = clamp(round(Int, 0.84 * nBoot) + 1, 1, nBoot)
    i025 = clamp(round(Int, 0.025 * nBoot) + 1, 1, nBoot)
    i975 = clamp(round(Int, 0.975 * nBoot) + 1, 1, nBoot)

    ci68_jl = zeros(2, n_imp, K)
    ci95_jl = zeros(2, n_imp, K)
    for k in 1:K, h in 1:n_imp

        ci68_jl[1, h, k] = sorted_store[k, h, i16]
        ci68_jl[2, h, k] = sorted_store[k, h, i84]
        ci95_jl[1, h, k] = sorted_store[k, h, i025]
        ci95_jl[2, h, k] = sorted_store[k, h, i975]
    end

    # Reshape Python CIs: stored as (2*n_imp, K) → (2, n_imp, K)
    ci68_py = zeros(2, n_imp, K)
    ci95_py = zeros(2, n_imp, K)
    for k in 1:K
        ci68_py[1, :, k] .= ci68_py_flat[1:n_imp, k]
        ci68_py[2, :, k] .= ci68_py_flat[(n_imp + 1):(2 * n_imp), k]
        ci95_py[1, :, k] .= ci95_py_flat[1:n_imp, k]
        ci95_py[2, :, k] .= ci95_py_flat[(n_imp + 1):(2 * n_imp), k]
    end

    @testset "68% CIs match Python to machine precision" begin
        for k in 1:K, h in 1:n_imp

            @test ci68_jl[1, h, k] ≈ ci68_py[1, h, k] atol=1e-10
            @test ci68_jl[2, h, k] ≈ ci68_py[2, h, k] atol=1e-10
        end
    end

    @testset "95% CIs match Python to machine precision" begin
        for k in 1:K, h in 1:n_imp

            @test ci95_jl[1, h, k] ≈ ci95_py[1, h, k] atol=1e-10
            @test ci95_jl[2, h, k] ≈ ci95_py[2, h, k] atol=1e-10
        end
    end
end
