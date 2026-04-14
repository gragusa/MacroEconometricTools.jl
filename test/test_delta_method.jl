# Test delta method standard errors for IRFs
using MacroEconometricTools
using Test
using LinearAlgebra
using Random
using Statistics
using StableRNGs: StableRNG

const IRFResult = MacroEconometricTools.IRFResult

@testset "Delta Method IRF Standard Errors" begin
    # Generate simple test data
    Random.seed!(456)
    T = 150
    n_v = 3
    n_l = 2

    # Generate AR(2) data with stronger signal
    Y = randn(T, n_v)
    for t in 3:T
        Y[t, :] = 0.6 * Y[t - 1, :] + 0.3 * Y[t - 2, :] + 0.1 * randn(n_v)
    end

    # Estimate VAR
    var = fit(OLSVAR, Y, n_l; names = [:Y1, :Y2, :Y3])

    @testset "Matrix Utilities" begin
        # Test duplication matrix
        D = MacroEconometricTools.duplication_matrix(3)
        @test size(D) == (9, 6)

        # Test elimination matrix
        L = MacroEconometricTools.elimination_matrix(3)
        @test size(L) == (6, 9)

        # Test commutation matrix
        K = MacroEconometricTools.commutation_matrix(3, 3)
        @test size(K) == (9, 9)

        # Test property: L * D = I
        @test L * D ≈ I(6) rtol=1e-10
    end

    @testset "Variance Computations" begin
        # Test coefficient covariance
        Σ_α = MacroEconometricTools.coefficient_covariance(var)
        @test size(Σ_α) == (n_v * n_l * n_v, n_v * n_l * n_v)
        @test all(isfinite.(Σ_α))
        # Check symmetry with tolerance for numerical precision
        @test maximum(abs.(Σ_α - Σ_α')) < 1e-10

        # Test sigma covariance
        Σ_σ = MacroEconometricTools.sigma_covariance(var)
        n_vech = n_v * (n_v + 1) ÷ 2
        @test size(Σ_σ) == (n_vech, n_vech)
        @test all(isfinite.(Σ_σ))
        # Check symmetry with tolerance
        @test maximum(abs.(Σ_σ - Σ_σ')) < 1e-10
    end

    @testset "IRF with Delta Method" begin
        # Compute IRF with delta method
        horizon = 12
        id = CholeskyID()

        irf_delta = irf(var, id; horizon = horizon, inference = Analytic())

        # Check structure
        @test irf_delta isa IRFResult
        # AxisArray layout is (variable, shock, horizon)
        @test size(irf_delta.irf) == (n_v, n_v, horizon + 1)
        @test size(irf_delta.stderr) == (n_v, n_v, horizon + 1)
        @test length(irf_delta.lower) == 3  # Default coverage levels
        @test length(irf_delta.upper) == 3

        # Check all elements are finite
        @test all(isfinite.(Array(irf_delta.irf)))
        @test all(isfinite.(Array(irf_delta.stderr)))

        # Standard errors should be positive
        @test all(Array(irf_delta.stderr) .>= 0)

        # Confidence bands should bracket the point estimate
        for i in 1:length(irf_delta.coverage)
            @test all(Array(irf_delta.lower[i]) .<= Array(irf_delta.irf))
            @test all(Array(irf_delta.upper[i]) .>= Array(irf_delta.irf))
        end

        # Standard errors should generally increase with horizon
        # (This is a statistical property, not guaranteed, so we just check trend)
        irf_stderr_data = Array(irf_delta.stderr)  # (n_v, n_v, horizon+1)
        avg_stderr_by_horizon = [mean(irf_stderr_data[:, :, h]) for h in 1:(horizon + 1)]
        # At least monotonically non-decreasing for first few horizons
        @test all(diff(avg_stderr_by_horizon[1:5]) .>= -1e-10)
    end

    @testset "Compare Delta vs Bootstrap" begin
        # Compute with both methods
        horizon = 8
        id = CholeskyID()

        irf_delta = irf(var, id; horizon = horizon, inference = Analytic())
        irf_boot = irf(var, id; horizon = horizon, inference = WildBootstrap(100),
            rng = StableRNG(789))

        # Both should give similar point estimates (same identification)
        @test Array(irf_delta.irf) ≈ Array(irf_boot.irf) rtol=1e-10

        # Just check both methods produce finite standard errors
        # We can't reliably compare magnitudes with low bootstrap reps
        @test all(isfinite.(Array(irf_boot.stderr)))
        @test all(irf_boot.stderr .>= 0)  # Allow zero at impact horizon
        @test all(isfinite.(irf_delta.stderr))
        @test all(irf_delta.stderr .>= 0)  # Allow zero at impact horizon
    end

    @testset "Delta Method Jacobian Matrices" begin
        # Test Jacobian computation
        horizon = 6
        id = CholeskyID()
        P = rotation_matrix(var, id)
        irf_point = MacroEconometricTools.compute_irf_point(var, P, horizon)

        G_matrices = MacroEconometricTools.irf_jacobian_matrices(var, irf_point, horizon)

        # Check we got the right number of matrices
        @test length(G_matrices) == horizon

        # Check dimensions
        for h in 1:horizon
            @test size(G_matrices[h]) == (n_v^2, n_l * n_v^2)
            @test all(isfinite.(G_matrices[h]))
        end
    end

    @testset "Delta Method Effect Covariance" begin
        horizon = 6
        id = CholeskyID()
        P = rotation_matrix(var, id)
        irf_point = MacroEconometricTools.compute_irf_point(var, P, horizon)

        V = MacroEconometricTools.irf_effect_covariance(var, P, irf_point)

        # Check structure
        @test size(V) == (horizon + 1, n_v^2, n_v^2)
        @test all(isfinite.(V))

        # Each slice should be symmetric (with tolerance)
        for h in 1:(horizon + 1)
            V_h = V[h, :, :]
            @test maximum(abs.(V_h - V_h')) < 1e-8
        end

        # First horizon (impact) should be zero (no uncertainty from coefficients)
        # Actually it has uncertainty from Σ, so just check it's small
        @test maximum(abs.(V[1, :, :])) < maximum(abs.(V[end, :, :]))
    end

    println("✓ All delta method tests passed!")
end
