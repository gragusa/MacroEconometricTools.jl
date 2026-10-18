using LinearAlgebra
using MacroEconometricTools
using MacroEconometricTools: StatsBase
using MacroEconometricTools.StatsBase: confint, stderror
using Test

@testset "StatsAPI methods match direct OLS formulas" begin
    fixture = correctness_fixture()
    model = fixture.model
    reference = reference_ols(fixture.sample, 2)
    (; X, coefficients, covariance) = reference
    n_obs, n_coef = size(X)
    n = 2

    @test coef(model) ≈ coefficients atol = 1.0e-12
    @test size(coef(model)) == (1 + 2n, n)
    @test StatsBase.modelmatrix(model) ≈ X
    @test fitted(model) ≈ X * coefficients atol = 1.0e-12
    @test fitted(model) + residuals(model) ≈ fixture.sample[3:end, :] atol = 1.0e-12
    @test Matrix(residual_covariance(model)) ≈ covariance atol = 1.0e-12

    @test StatsBase.nobs(model) == size(fixture.sample, 1) - 2 == n_obs
    @test StatsBase.dof_residual(model) == n_obs - n_coef
    @test StatsBase.dof(model) == n * n_coef

    # Covariance of vec(B): equation j, regressor i has variance Σ[j,j] (X'X)⁻¹[i,i],
    # and the cross-equation block (j, k) is Σ[j,k] (X'X)⁻¹.
    XXinv = inv(X' * X)
    V = vcov(model)
    @test size(V) == (n * n_coef, n * n_coef)
    @test V ≈ kron(covariance, XXinv) rtol = 1.0e-10
    @test V[1:n_coef, (n_coef + 1):end] ≈ covariance[1, 2] * XXinv rtol = 1.0e-10
    se = stderror(model)
    @test size(se) == size(coef(model))
    for j in 1:n, i in 1:n_coef

        @test se[i, j] ≈ sqrt(covariance[j, j] * XXinv[i, i]) rtol = 1.0e-10
    end

    ci = confint(model)
    @test size(ci) == (length(coef(model)), 2)
    z95 = 1.959963984540054
    @test ci[:, 1] ≈ vec(coefficients) .- z95 .* vec(se) rtol = 1.0e-10
    @test ci[:, 2] ≈ vec(coefficients) .+ z95 .* vec(se) rtol = 1.0e-10
    z90 = 1.6448536269514722
    @test confint(model; level = 0.9)[:, 2] ≈ vec(coefficients) .+ z90 .* vec(se) rtol = 1.0e-10
    @test_throws "level must be in (0, 1)" confint(model; level = 1.5)

    ll = -n_obs / 2 * (n * log(2π) + logdet(covariance) + n)
    k = n * n_coef
    @test loglikelihood(model) ≈ ll rtol = 1.0e-12
    @test aic(model) ≈ -2ll + 2k rtol = 1.0e-12
    @test bic(model) ≈ -2ll + k * log(n_obs) rtol = 1.0e-12
    @test hqic(model) ≈ -2ll + 2k * log(log(n_obs)) rtol = 1.0e-12
end
