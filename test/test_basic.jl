# Basic smoke tests for MacroEconometricTools
using MacroEconometricTools
using Test
using LinearAlgebra
using Random
using StableRNGs: StableRNG

@testset "Basic Package Functionality" begin

    # Generate simple test data
    Random.seed!(123)
    T = 100
    n_v = 3
    n_l = 2

    # Generate AR(2) data
    Y = randn(T, n_v)
    for t in 3:T
        Y[t, :] = 0.5 * Y[t - 1, :] + 0.3 * Y[t - 2, :] + 0.1 * randn(n_v)
    end

    @testset "VAR Estimation" begin
        # Basic OLS VAR
        var = fit(OLSVAR, Y, n_l)

        @test var isa VARModel{Float64, OLSVAR}
        @test n_vars(var) == n_v
        @test n_lags(var) == n_l
        @test effective_obs(var) == T - n_l

        # Check coefficients
        coefs = coef(var)
        @test length(coefs.intercept) == n_v
        @test size(coefs.lags) == (n_v, n_v, n_l)

        # Check residuals
        resid = residuals(var)
        @test size(resid) == (T - n_l, n_v)

        # Check covariance
        Σ = vcov(var)
        @test size(Σ) == (n_v, n_v)
        @test Σ isa Symmetric
    end

    @testset "VAR with Constraints" begin
        # Test zero constraint
        constraints = [ZeroConstraint(:Y_1, [:Y_2], [1])]
        var_constrained = fit(OLSVAR, Y, n_l; constraints = constraints)

        @test var_constrained isa VARModel
        @test !isnothing(var_constrained.coefficients.constraints)
    end

    @testset "Identification" begin
        var = fit(OLSVAR, Y, n_l)

        # Cholesky identification
        id = CholeskyID()
        P = rotation_matrix(var, id)

        @test size(P) == (n_v, n_v)
        @test istril(P)  # Should be lower triangular

        # Check that P*P' ≈ Σ
        Σ = vcov(var)
        @test P * P' ≈ Matrix(Σ) rtol=1e-10
    end

    @testset "Resampling reproducibility" begin
        var = fit(OLSVAR, Y, n_l)
        id = CholeskyID()

        boot1 = bootstrap_irf(var, id, 5, 20; rng = StableRNG(20240612))
        boot2 = bootstrap_irf(var, id, 5, 20; rng = StableRNG(20240612))
        @test boot1 == boot2
    end

    @testset "Utility Functions" begin
        # Test lag function
        x = collect(1.0:10.0)
        x_lag = lag(x, 1)
        @test isnan(x_lag[1])  # Now returns NaN instead of missing
        @test x_lag[2] == 1.0
        @test x_lag[end] == 9.0

        # Test companion form
        A = randn(n_v, n_v, n_l)
        F = companion_form(A)
        @test size(F) == (n_v * n_l, n_v * n_l)
    end

    println("✓ All basic tests passed!")
end
