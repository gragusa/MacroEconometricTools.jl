using Test
using LinearAlgebra
using Statistics: mean
using MacroEconometricTools

@testset "Independent OLS and IRF oracles" begin
    fixture = correctness_fixture()
    model = fixture.model
    reference = reference_ols(fixture.sample, 2)

    @testset "estimator equals direct least squares" begin
        @test coef(model) ≈ reference.coefficients atol = 1.0e-12
        estimated = model.coefficients
        @test estimated.intercept ≈ vec(reference.coefficients[1, :]) atol = 1.0e-12
        # OLS rows are (variable, lag) and columns are equations; the model
        # stores (equation, variable, lag).
        @test estimated.lags ≈ permutedims(
            reshape(reference.coefficients[2:end, :], 2, 2, 2), (3, 1, 2)
        ) atol = 1.0e-12
        @test residuals(model) ≈ reference.innovations atol = 1.0e-12
        @test Matrix(residual_covariance(model)) ≈ reference.covariance atol = 1.0e-12
    end

    @testset "IRF equals direct companion recursion" begin
        alpha = vec(reshape(model.coefficients.lags, 2, 4))
        covariance = Matrix(residual_covariance(model))
        sigma_half = [covariance[1, 1], covariance[2, 1], covariance[2, 2]]
        expected = reference_irf(alpha, sigma_half, 2, 2, 8)
        actual = permutedims(Array(irf(model, CholeskyID(); horizon = 8).irf), (3, 1, 2))
        @test actual ≈ expected atol = 2.0e-12
    end
end

@testset "Analytic IRF standard errors against numerical differentiation" begin
    model = correctness_fixture().model
    n, p, horizon = n_vars(model), n_lags(model), 6
    alpha = vec(reshape(model.coefficients.lags, n, n * p))
    covariance = Matrix(residual_covariance(model))
    sigma_half = [covariance[row, column] for column in 1:n for row in column:n]
    parameters = [alpha; sigma_half]

    parameter_covariance = cat(
        MacroEconometricTools.coefficient_covariance(model),
        MacroEconometricTools.sigma_covariance(model); dims = (1, 2)
    )
    irf_map = θ -> reference_irf(
        view(θ, eachindex(alpha)),
        view(θ, (length(alpha) + 1):length(θ)), n, p, horizon
    )
    jacobian = central_jacobian(irf_map, parameters)
    variance = jacobian * parameter_covariance * jacobian'
    expected = reshape(sqrt.(max.(diag(variance), 0.0)), horizon + 1, n, n)

    result = irf(model, CholeskyID(); horizon, inference = Analytic())
    actual = permutedims(Array(result.stderr), (3, 1, 2))
    @test actual ≈ expected rtol = 2.0e-5 atol = 2.0e-8
    @test all(diag(actual[1, :, :]) .> 0)
    @test actual[1, 2, 1] > 0
    @test actual[1, 1, 2] == 0

    coverage_index = findfirst(==(0.95), result.coverage)
    z = 1.959963984540054
    point, stderr = Array(result.irf), Array(result.stderr)
    @test Array(result.lower[coverage_index]) ≈ point .- z .* stderr rtol = 1.0e-12
    @test Array(result.upper[coverage_index]) ≈ point .+ z .* stderr rtol = 1.0e-12
end

@testset "Variance decomposition against squared reference IRFs" begin
    model = correctness_fixture().model
    n, horizon = n_vars(model), 8
    alpha = vec(reshape(model.coefficients.lags, n, 2n))
    covariance = Matrix(residual_covariance(model))
    sigma_half = [covariance[1, 1], covariance[2, 1], covariance[2, 2]]
    responses = reference_irf(alpha, sigma_half, n, 2, horizon)  # (horizon, var, shock)

    expected = zeros(n, n, horizon + 1)
    for h in 1:(horizon + 1), i in 1:n

        total = sum(responses[1:h, i, :] .^ 2)
        for j in 1:n
            expected[i, j, h] = sum(responses[1:h, i, j] .^ 2) / total
        end
    end

    result = irf(model, CholeskyID(); horizon)
    fevd = variance_decomposition(result)
    @test size(fevd) == (n, n, horizon + 1)
    @test Array(fevd) ≈ expected atol = 1.0e-12
    @test all(sum(Array(fevd); dims = 2) .≈ 1)
    # Cholesky ordering: the second shock does not move the first variable on impact.
    @test fevd[1, 2, 1] == 0
    @test fevd[:output, :price, 3] == expected[1, 2, 3]
    @test Array(variance_decomposition(result; horizon = 4)) ≈ expected[:, :, 4]

    @test_throws "horizon must be in 1:9" variance_decomposition(result; horizon = 0)
    @test_throws "unit-variance shocks" variance_decomposition(
        irf(model, CholeskyID(); horizon, normalization = UnitEffect())
    )
    rescaled = rescale(result, :price => 2.0)
    @test_throws "unscaled IRFs" variance_decomposition(rescaled)
end

@testset "Bootstrap draws share the IRF layout" begin
    model = correctness_fixture().model
    n, horizon, reps = n_vars(model), 5, 20
    point = Array(irf(model, CholeskyID(); horizon).irf)
    # The default ordering takes the fast Cholesky path; an explicit ordering
    # takes the generic refit path.
    for id in (CholeskyID(), CholeskyID([:output, :price]))
        draws = bootstrap_irf(model, id, WildBootstrap(reps); horizon, rng = StableRNG(1))
        @test size(draws) == (reps, n, n, horizon + 1)
        # (draw, variable, shock, horizon): no impact response of variable 1 to shock 2
        @test all(iszero, draws[:, 1, 2, 1])
        @test all(>(0), draws[:, 2, 2, 1])
        @test dropdims(mean(draws; dims = 1); dims = 1) ≈ point rtol = 0.5 atol = 0.05
    end
end
