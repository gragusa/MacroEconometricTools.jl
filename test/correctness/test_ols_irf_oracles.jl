using Test
using LinearAlgebra
using MacroEconometricTools

@testset "Independent OLS and IRF oracles" begin
    fixture = correctness_fixture()
    model = fixture.model
    reference = reference_ols(fixture.sample, 2)

    @testset "estimator equals direct least squares" begin
        estimated = coef(model)
        @test estimated.intercept ≈ vec(reference.coefficients[1, :]) atol = 1e-12
        # OLS rows are (variable, lag) and columns are equations; the model
        # stores (equation, variable, lag).
        @test estimated.lags ≈ permutedims(
            reshape(reference.coefficients[2:end, :], 2, 2, 2), (3, 1, 2)) atol = 1e-12
        @test residuals(model) ≈ reference.innovations atol = 1e-12
        @test Matrix(vcov(model)) ≈ reference.covariance atol = 1e-12
    end

    @testset "IRF equals direct companion recursion" begin
        alpha = vec(reshape(coef(model).lags, 2, 4))
        covariance = Matrix(vcov(model))
        sigma_half = [covariance[1, 1], covariance[2, 1], covariance[2, 2]]
        expected = reference_irf(alpha, sigma_half, 2, 2, 8)
        actual = permutedims(Array(irf(model, CholeskyID(); horizon = 8).irf), (3, 1, 2))
        @test actual ≈ expected atol = 2e-12
    end
end

@testset "Analytic IRF standard errors against numerical differentiation" begin
    model = correctness_fixture().model
    n, p, horizon = n_vars(model), n_lags(model), 6
    alpha = vec(reshape(coef(model).lags, n, n * p))
    covariance = Matrix(vcov(model))
    sigma_half = [covariance[row, column] for column in 1:n for row in column:n]
    parameters = [alpha; sigma_half]

    parameter_covariance = cat(MacroEconometricTools.coefficient_covariance(model),
        MacroEconometricTools.sigma_covariance(model); dims = (1, 2))
    irf_map = θ -> reference_irf(view(θ, eachindex(alpha)),
        view(θ, (length(alpha) + 1):length(θ)), n, p, horizon)
    jacobian = central_jacobian(irf_map, parameters)
    variance = jacobian * parameter_covariance * jacobian'
    expected = reshape(sqrt.(max.(diag(variance), 0.0)), horizon + 1, n, n)

    result = irf(model, CholeskyID(); horizon, inference = Analytic())
    actual = permutedims(Array(result.stderr), (3, 1, 2))
    @test actual≈expected rtol=2e-5 atol=2e-8
    @test all(diag(actual[1, :, :]) .> 0)
    @test actual[1, 2, 1] > 0
    @test actual[1, 1, 2] == 0

    coverage_index = findfirst(==(0.95), result.coverage)
    z = 1.959963984540054
    point, stderr = Array(result.irf), Array(result.stderr)
    @test Array(result.lower[coverage_index]) ≈ point .- z .* stderr rtol = 1e-12
    @test Array(result.upper[coverage_index]) ≈ point .+ z .* stderr rtol = 1e-12
end
