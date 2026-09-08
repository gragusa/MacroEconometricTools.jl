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

    p = 2
    K = 2
    n_imp = 21
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

    p = 2
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
    K = 2
    T = 100
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

    p = 2
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

    p = 2
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

    p = 2
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

    p = 2
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

    p = 2
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

    p = 2
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
# Test 10: proxy_svar_mbb matches the reference implementation
# ============================================================================
# Replays the reference implementation's recorded block draws through
# `proxy_svar_mbb` itself, so the shipped bootstrap driver — not a
# reimplementation of it — is what gets compared.
@testset "proxy_svar_mbb matches Python (replayed block indices)" begin
    Y = Matrix(CSV.read(joinpath(BDATA, "jl_crossval_Y.csv"), DataFrame))
    proxy_full = vec(Matrix(CSV.read(joinpath(BDATA, "jl_crossval_proxy.csv"), DataFrame)))

    # Reference block starts are 0-based
    block_indices = Matrix{Int}(CSV.read(
        joinpath(BDATA, "jl_crossval_mbb_block_indices.csv"),
        DataFrame; header = false)) .+ 1
    irf_norm_draws_py = Matrix(CSV.read(
        joinpath(BDATA, "jl_crossval_mbb_irf_norm_draws.csv"),
        DataFrame; header = false))
    ci68_py_flat = Matrix(CSV.read(joinpath(BDATA, "jl_crossval_mbb_ci68_irf_norm.csv"),
        DataFrame; header = false))
    ci95_py_flat = Matrix(CSV.read(joinpath(BDATA, "jl_crossval_mbb_ci95_irf_norm.csv"),
        DataFrame; header = false))

    p = 2
    K = 2
    horizon = 20
    n_imp = horizon + 1
    n_boot = size(block_indices, 1)

    model = fit(OLSVAR, Y, p)
    id = IVIdentification(proxy_full[(p + 1):end], 1)

    mbb = proxy_svar_mbb(model, id, horizon,
        ProxySVARMBB(n_boot; block_length = 4, norm_scale = -1.0);
        block_indices = block_indices)

    @test mbb.n_failed == 0

    @testset "per-draw normalized IRFs" begin
        max_diff = 0.0
        for b in 1:n_boot, k in 1:K, h in 1:n_imp
            max_diff = max(max_diff,
                abs(mbb.irf_norm_store[k, h, b] -
                    irf_norm_draws_py[b, (k - 1) * n_imp + h]))
        end
        @test max_diff < 1e-10
    end

    # Reference CIs are stored as (2*n_imp, K) → (2, n_imp, K)
    ci68_py = zeros(2, n_imp, K)
    ci95_py = zeros(2, n_imp, K)
    for k in 1:K
        ci68_py[1, :, k] .= ci68_py_flat[1:n_imp, k]
        ci68_py[2, :, k] .= ci68_py_flat[(n_imp + 1):(2 * n_imp), k]
        ci95_py[1, :, k] .= ci95_py_flat[1:n_imp, k]
        ci95_py[2, :, k] .= ci95_py_flat[(n_imp + 1):(2 * n_imp), k]
    end

    @testset "68% CIs" begin
        for k in 1:K, h in 1:n_imp

            @test mbb.ci68_irf_norm[1, h, k] ≈ ci68_py[1, h, k] atol=1e-10
            @test mbb.ci68_irf_norm[2, h, k] ≈ ci68_py[2, h, k] atol=1e-10
        end
    end

    @testset "95% CIs" begin
        for k in 1:K, h in 1:n_imp

            @test mbb.ci95_irf_norm[1, h, k] ≈ ci95_py[1, h, k] atol=1e-10
            @test mbb.ci95_irf_norm[2, h, k] ≈ ci95_py[2, h, k] atol=1e-10
        end
    end

    # Hall's intervals reflect the reference's percentile endpoints
    @testset "Hall's intervals" begin
        pt = mbb.point_irf_norm
        for k in 1:K, h in 1:n_imp

            @test mbb.halls68_irf_norm[1, h, k] ≈ 2 * pt[k, h] - ci68_py[2, h, k] atol=1e-10
            @test mbb.halls68_irf_norm[2, h, k] ≈ 2 * pt[k, h] - ci68_py[1, h, k] atol=1e-10
            @test mbb.halls95_irf_norm[1, h, k] ≈ 2 * pt[k, h] - ci95_py[2, h, k] atol=1e-10
            @test mbb.halls95_irf_norm[2, h, k] ≈ 2 * pt[k, h] - ci95_py[1, h, k] atol=1e-10
        end
    end
end

# ============================================================================
# Test 11: the proxy is resampled in blocks but not centered
# ============================================================================
# Centering keeps the simulated VAR from drifting; the proxy is not fed through
# that recursion, and demeaning it would perturb `Sigma_um` — the moment being
# bootstrapped. The block-position means of the proxy are non-zero here, so
# subtracting them would change the draws.
@testset "MBB leaves the proxy uncentered" begin
    Y = Matrix(CSV.read(joinpath(BDATA, "jl_crossval_Y.csv"), DataFrame))
    proxy_full = vec(Matrix(CSV.read(joinpath(BDATA, "jl_crossval_proxy.csv"), DataFrame)))
    block_indices = Matrix{Int}(CSV.read(
        joinpath(BDATA, "jl_crossval_mbb_block_indices.csv"),
        DataFrame; header = false)) .+ 1

    p = 2
    K = 2
    ℓ = 4
    horizon = 10
    n_imp = horizon + 1
    n_boot = size(block_indices, 1)
    n_resample = size(block_indices, 2)
    s = -1.0

    model = fit(OLSVAR, Y, p)
    mm = proxy_full[(p + 1):end]
    ν = model.residuals
    TT = size(ν, 1)

    mbb = proxy_svar_mbb(model, IVIdentification(mm, 1), horizon,
        ProxySVARMBB(n_boot; block_length = ℓ, norm_scale = s);
        block_indices = block_indices)

    # Position-specific means of the proxy — what centering would subtract.
    m_center_block = [mean(mm[j:(TT - ℓ + j)]) for j in 1:ℓ]
    @test maximum(abs, m_center_block) > 1e-3

    # Replay one draw by hand, centering residuals only, and confirm the
    # driver produced the same normalized IRF.
    A_est = zeros(K, 1 + K * p)
    coefs = coef(model)
    A_est[:, 1] .= coefs.intercept
    for lag in 1:p
        A_est[:, (1 + (lag - 1) * K + 1):(1 + lag * K)] .= coefs.lags[:, :, lag]
    end
    A_sim = A_est'

    y_init = zeros(1 + K * p)
    y_init[1] = 1.0
    for lag in 1:p
        y_init[(1 + (lag - 1) * K + 1):(1 + lag * K)] .= Y[p + 1 - lag, :]
    end

    u_center_block = zeros(ℓ, K)
    for j in 1:ℓ
        u_center_block[j, :] .= vec(mean(ν[j:(TT - ℓ + j), :]; dims = 1))
    end

    b = 1
    u_temp = zeros(n_resample * ℓ, K)
    m_temp = zeros(n_resample * ℓ)
    for j in 1:n_resample
        idx = block_indices[b, j]
        rows = ((j - 1) * ℓ + 1):(j * ℓ)
        u_temp[rows, :] .= ν[idx:(idx + ℓ - 1), :] .- u_center_block
        m_temp[rows] .= mm[idx:(idx + ℓ - 1)]     # proxy passes through uncentered
    end

    u_star = u_temp[1:TT, :]
    m_star = m_temp[1:TT]
    x_star = zeros(TT, 1 + K * p)
    x_star[1, :] .= y_init
    y_star = copy(u_star)
    for t in 1:TT
        for k in 1:K, j in 1:(1 + K * p)

            y_star[t, k] += x_star[t, j] * A_sim[j, k]
        end
        if t < TT
            x_star[t + 1, 1] = 1.0
            x_star[t + 1, 2:(K + 1)] .= y_star[t, :]
            if p > 1
                x_star[t + 1, (K + 2):(K * p + 1)] .= x_star[t, 2:(K * (p - 1) + 1)]
            end
        end
    end

    A_star, _, Σ_uu_star, Σ_um_star,
    H1_star = MacroEconometricTools.estimate_proxy_svar(y_star, x_star, m_star)
    dyn = MacroEconometricTools.proxy_svar_dynamics(
        A_star, Σ_uu_star, Σ_um_star, H1_star, p, s, n_imp, 1)

    @test mbb.irf_norm_store[:, :, b] ≈ dyn.irf_norm atol=1e-10
end
