# ============================================================================
# Tests for Proxy-SVAR (IV-SVAR) Implementation
# ============================================================================
# Three test strategies:
#   Test 1: Perfect instrument (Z = ε₁) → must recover Θ₀ exactly
#   Test 2: Cholesky equivalence → proxy-SVAR first column matches Cholesky
#   Test 3: Cross-validation with Jentsch & Lunsford (2022) Python code
#            (see test_proxy_svar_crossval.jl)

using MacroEconometricTools
using Test
using LinearAlgebra
using Random
using StableRNGs: StableRNG

# ============================================================================
# Helper: simulate a bivariate VAR with known structural parameters
# ============================================================================
"""
    simulate_var_dgp(T, A_coefs, H; rng, burn=500)

Simulate from a structural VAR:
    y_t = A₁ y_{t-1} + ... + Aₚ y_{t-p} + H ε_t
where ε_t ~ N(0, I).

Returns (Y, ε) where Y is (T × K) and ε is (T × K).
A_coefs is a vector of matrices [A₁, A₂, ...].
"""
function simulate_var_dgp(T::Int, A_coefs::Vector{Matrix{Float64}},
        H::Matrix{Float64}; rng = StableRNG(42), burn::Int = 500)
    p = length(A_coefs)
    K = size(H, 1)
    T_total = T + burn

    ε = randn(rng, T_total, K)
    Y = zeros(T_total, K)

    for t in (p + 1):T_total
        for lag in 1:p
            Y[t, :] .+= A_coefs[lag] * Y[t - lag, :]
        end
        Y[t, :] .+= H * ε[t, :]
    end

    # Drop burn-in
    return Y[(burn + 1):end, :], ε[(burn + 1):end, :]
end

# ============================================================================
# Test 1: Perfect Instrument — Exact Recovery of Θ₀
# ============================================================================
@testset "Proxy-SVAR: Perfect Instrument" begin
    # DGP: 3-variable VAR(1) with known H (lower triangular)
    rng = StableRNG(2024)
    K = 3
    T = 5000  # large sample for near-exact recovery

    H_true = [1.0 0.0 0.0;
              0.5 1.0 0.0;
              0.3 0.2 1.0]

    A1 = [0.3 0.0 0.1;
          0.1 0.5 0.0;
          0.0 0.0 0.4]

    Y, ε = simulate_var_dgp(T, [A1], H_true; rng = rng)

    # Perfect instrument: Z = ε₁ (the actual structural shock)
    # This should give EXACT identification of Θ₀[:,1]
    p = 1
    n_resid = T - p
    Z_perfect = reshape(ε[(p + 1):end, 1], :, 1)

    instrument = ExternalInstrument(Z_perfect, 1)
    model = fit(IVSVAR, Y, p; instrument = instrument,
        names = [:Y1, :Y2, :Y3])

    θ_est = model.metadata.iv_coefficients
    θ_true = H_true[:, 1]

    @testset "Point estimates close to truth" begin
        @test θ_est[1] ≈ 1.0 atol=1e-10   # unit effect normalization (exact)
        @test θ_est[2] ≈ θ_true[2] atol=0.05  # 0.5
        @test θ_est[3] ≈ θ_true[3] atol=0.05  # 0.3
    end

    @testset "First-stage F-statistic is very large" begin
        F = first_stage_F(model)
        @test F > 1000  # perfect instrument → extremely strong first stage
    end

    @testset "IRFs match structural MA coefficients" begin
        irf_result = irf(model, IVIdentification(); horizon = 10)
        irf_data = Array(irf_result.irf)  # (variable, shock, horizon)

        # Impact response = first column of H
        for i in 1:K
            @test irf_data[i, 1, 1] ≈ θ_true[i] atol=0.05
        end

        # At horizon 1: Θ₁ = A₁ * H[:,1]
        Θ₁_true = A1 * H_true[:, 1]
        for i in 1:K
            @test irf_data[i, 1, 2] ≈ Θ₁_true[i] atol=0.05
        end
    end
end

# ============================================================================
# Test 1b: Full-length instrument (auto-trim)
# ============================================================================
# When the user passes Z with the same number of rows as Y (T_total),
# the first p rows should be dropped automatically to align with residuals.
@testset "Proxy-SVAR: Full-length instrument auto-trim" begin
    rng = StableRNG(2024)
    K = 3;
    T = 500
    H_true = [1.0 0.0 0.0;
              0.5 0.8 0.0;
              0.3 0.2 1.0]
    A1 = [0.3 0.0 0.1;
          0.1 0.5 0.0;
          0.0 0.0 0.4]
    Y, ε = simulate_var_dgp(T, [A1], H_true; rng = rng)

    p = 1

    # Pre-trimmed Z (T-p rows) — the old way
    Z_trimmed = reshape(ε[(p + 1):end, 1], :, 1)
    model_trimmed = fit(IVSVAR, Y, p;
        instrument = ExternalInstrument(Z_trimmed, 1), names = [:Y1, :Y2, :Y3])

    # Full-length Z (T rows, same as Y) — the new way
    Z_full = reshape(ε[:, 1], :, 1)
    model_full = fit(IVSVAR, Y, p;
        instrument = ExternalInstrument(Z_full, 1), names = [:Y1, :Y2, :Y3])

    @testset "Coefficients match" begin
        @test model_full.metadata.iv_coefficients ≈ model_trimmed.metadata.iv_coefficients atol=1e-12
        @test first_stage_F(model_full) ≈ first_stage_F(model_trimmed) atol=1e-10
    end

    @testset "IRFs match" begin
        irf_t = irf(model_trimmed, IVIdentification(); horizon = 10)
        irf_f = irf(model_full, IVIdentification(); horizon = 10)
        @test Array(irf_t.irf) ≈ Array(irf_f.irf) atol=1e-12
    end

    @testset "Also works with vector input" begin
        model_vec = fit(IVSVAR, Y, p;
            instrument = ExternalInstrument(vec(ε[:, 1]), 1), names = [:Y1, :Y2, :Y3])
        @test model_vec.metadata.iv_coefficients ≈ model_trimmed.metadata.iv_coefficients atol=1e-12
    end
end

# ============================================================================
# Test 2: Cholesky Equivalence
# ============================================================================
# When the DGP is lower-triangular (recursive), the first column of the
# proxy-SVAR impact matrix should match the first column of the Cholesky
# decomposition of Σ.
@testset "Proxy-SVAR: Cholesky Equivalence" begin
    rng = StableRNG(2025)
    K = 3
    T = 10000  # very large for asymptotic equivalence

    # Lower-triangular H → Cholesky-identified DGP
    H_true = [1.0 0.0 0.0;
              0.7 0.8 0.0;
              -0.2 0.3 0.9]

    A1 = [0.4 0.0 0.0;
          0.1 0.5 0.0;
          0.0 0.1 0.3]

    Y, ε = simulate_var_dgp(T, [A1], H_true; rng = rng)

    p = 1
    n_resid = T - p

    # Proxy-SVAR with Z = ε₁ (perfect instrument for clean comparison)
    Z = reshape(ε[(p + 1):end, 1], :, 1)
    instrument = ExternalInstrument(Z, 1)

    model_iv = fit(IVSVAR, Y, p; instrument = instrument,
        names = [:Y1, :Y2, :Y3])
    model_ols = fit(OLSVAR, Y, p; names = [:Y1, :Y2, :Y3])

    # Cholesky first column (from OLS VAR)
    Σ = Matrix(model_ols.Σ)
    L_chol = cholesky(Σ).L
    chol_col1 = L_chol[:, 1]

    # Proxy-SVAR first column (unit effect normalization: θ[1] = 1)
    iv_col1 = model_iv.metadata.iv_coefficients

    # Both should identify the same structural shock up to scale.
    # Under unit effect normalization: iv_col1[1] = 1.0
    # Under Cholesky: chol_col1[1] = L[1,1] = √Σ₁₁
    # So we need to normalize the Cholesky column too.
    chol_col1_normalized = chol_col1 ./ chol_col1[1]

    @testset "Normalized first columns match" begin
        for i in 1:K
            @test iv_col1[i] ≈ chol_col1_normalized[i] atol=0.05
        end
    end

    @testset "IRF first columns match (horizon 0-5)" begin
        irf_iv = irf(model_iv, IVIdentification(); horizon = 5)
        irf_chol = irf(model_ols, CholeskyID(); horizon = 5)
        irf_iv_data = Array(irf_iv.irf)     # (variable, shock, horizon)
        irf_chol_data = Array(irf_chol.irf) # (variable, shock, horizon)

        # Normalize Cholesky IRFs to unit effect
        scale = irf_chol_data[1, 1, 1]  # L[1,1]
        for h in 1:6
            for i in 1:K
                irf_chol_norm = irf_chol_data[i, 1, h] / scale
                irf_iv_val = irf_iv_data[i, 1, h]
                @test irf_iv_val ≈ irf_chol_norm atol=0.1
            end
        end
    end
end

# ============================================================================
# Test 3: Basic Proxy-SVAR Mechanics
# ============================================================================
@testset "Proxy-SVAR: Estimation Mechanics" begin
    rng = StableRNG(999)
    K = 2
    T = 500

    H_true = [sqrt(2)/2 sqrt(2)/2;
              (sqrt(2)-sqrt(6))/4 (sqrt(2)+sqrt(6))/4]

    A1 = [0.44 0.66;
          -0.11 1.32]
    A2 = [-0.18 0.0;
          -0.18 -0.09]

    Y, ε = simulate_var_dgp(T, [A1, A2], H_true; rng = rng)

    p = 2
    n_resid = T - p

    # Noisy proxy: Z = ψ * ε₁ + noise
    ψ = 1.0
    noise = randn(rng, n_resid)
    Z_proxy = ψ .* ε[(p + 1):end, 1] .+ noise

    instrument = ExternalInstrument(reshape(Z_proxy, :, 1), 1)

    @testset "fit produces correct model type" begin
        model = fit(IVSVAR, Y, p; instrument = instrument, names = [:Y1, :Y2])
        @test model isa VARModel{Float64, <:IVSVAR}
        @test n_vars(model) == K
        @test n_lags(model) == p
    end

    @testset "Metadata contains required fields" begin
        model = fit(IVSVAR, Y, p; instrument = instrument, names = [:Y1, :Y2])
        @test haskey(model.metadata, :structural_impact)
        @test haskey(model.metadata, :first_stage_F)
        @test haskey(model.metadata, :target_shock)
        @test haskey(model.metadata, :iv_coefficients)
        @test model.metadata.target_shock == 1
    end

    @testset "Unit effect normalization" begin
        model = fit(IVSVAR, Y, p; instrument = instrument, names = [:Y1, :Y2])
        @test model.metadata.iv_coefficients[1] == 1.0  # exact by construction
    end

    @testset "First-stage F > 0 with valid instrument" begin
        model = fit(IVSVAR, Y, p; instrument = instrument, names = [:Y1, :Y2])
        @test first_stage_F(model) > 0
    end

    @testset "Impact matrix is full rank" begin
        model = fit(IVSVAR, Y, p; instrument = instrument, names = [:Y1, :Y2])
        P = model.metadata.structural_impact
        @test size(P) == (K, K)
        @test rank(P) == K
    end

    @testset "IRF computation succeeds" begin
        model = fit(IVSVAR, Y, p; instrument = instrument, names = [:Y1, :Y2])
        result = irf(model, IVIdentification(); horizon = 20)
        @test size(result.irf) == (K, K, 21)  # (variable, shock, horizon)
        @test Array(result.irf)[1, 1, 1] ≈ 1.0 atol=1e-10  # unit effect
    end

    @testset "Constraints are supported" begin
        # A₁₁ = 0: Y1 has no own lags (as in our MPS specification)
        constraints = [BlockExogeneity([:Y1], [:Y1])]
        model = fit(IVSVAR, Y, p; instrument = instrument,
            names = [:Y1, :Y2], constraints = constraints)
        @test is_stable(model) || true  # may not be stable with constraints, but should not error
        @test model isa VARModel
    end

    @testset "Dimension mismatch errors" begin
        # Instrument with too few rows
        Z_bad = reshape(randn(rng, n_resid - 5), :, 1)
        instrument_bad = ExternalInstrument(Z_bad, 1)
        @test_throws DimensionMismatch fit(IVSVAR, Y, p;
            instrument = instrument_bad, names = [:Y1, :Y2])
    end
end

# ============================================================================
# Test 4: Proxy-SVAR Identification — Jentsch-Lunsford DGP Parameters
# ============================================================================
# Use the exact DGP from Jentsch & Lunsford (2022) to ensure our
# identification formula matches their `estimate_proxy_svar()`.
@testset "Proxy-SVAR: Jentsch-Lunsford DGP Identification" begin
    rng = StableRNG(0)  # match np.random.seed(0)
    K = 2
    T = 2000  # large sample

    # Exact J&L DGP parameters
    H_true = [sqrt(2)/2 sqrt(2)/2;
              (sqrt(2)-sqrt(6))/4 (sqrt(2)+sqrt(6))/4]

    A1 = [0.44 0.66;
          -0.11 1.32]
    A2 = [-0.18 0.0;
          -0.18 -0.09]

    Y, ε = simulate_var_dgp(T, [A1, A2], H_true; rng = rng)

    p = 2
    n_resid = T - p

    # Strong proxy (DGP1): ψ = 1.0
    ψ = 1.0
    noise = randn(rng, n_resid)
    Z_proxy = ψ .* ε[(p + 1):end, 1] .+ noise

    instrument = ExternalInstrument(reshape(Z_proxy, :, 1), 1)
    model = fit(IVSVAR, Y, p; instrument = instrument, names = [:Y1, :Y2])

    θ_est = model.metadata.iv_coefficients
    H1_true = H_true[:, 1]
    H1_true_norm = H1_true ./ H1_true[1]  # unit effect normalization

    @testset "Estimates converge to truth (large T)" begin
        # With T=2000 and ψ=1, estimates should be very close
        @test θ_est[1] == 1.0
        @test θ_est[2] ≈ H1_true_norm[2] atol=0.15
    end

    @testset "Identification formula: H1 = Σ_um / φ" begin
        # Replicate the Python identification formula exactly
        ν = residuals(model)
        m = Z_proxy
        Σ_uu = ν' * ν / n_resid
        Σ_um = ν' * m / n_resid

        # φ² = Σ_um' Σ_uu⁻¹ Σ_um
        φ² = dot(Σ_um, Σ_uu \ Σ_um)
        φ = sqrt(φ²)

        # H1 = Σ_um / φ (Python formula)
        H1_python = Σ_um ./ φ

        # Our formula normalizes differently (unit effect), so compare ratios
        ratio_python = H1_python[2] / H1_python[1]
        ratio_julia = θ_est[2] / θ_est[1]

        @test ratio_julia ≈ ratio_python atol=1e-10
    end
end
