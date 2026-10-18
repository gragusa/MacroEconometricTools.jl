# ============================================================================
# confidence_bands, forecast, historical_decomposition, and VAR(::DataFrame, ...)
# ============================================================================

using MacroEconometricTools
using Test
using Random
using LinearAlgebra: cholesky
using Statistics: cov, mean
using StableRNGs: StableRNG
using CSV
using DataFrames

function post_estimation_data(rng, T, n_v)
    Y = randn(rng, T, n_v)
    for t in 3:T
        Y[t, :] = 0.5 * Y[t - 1, :] + 0.2 * Y[t - 2, :] + 0.3 * randn(rng, n_v)
    end
    return Y
end

@testset "confidence_bands" begin
    Y = post_estimation_data(StableRNG(2024), 150, 3)
    model = fit(OLSVAR, Y, 2; names = [:a, :b, :c])
    horizon = 6
    boot(; coverage = [0.68, 0.9, 0.95], save_draws = true, kw...) = irf(
        model, CholeskyID(); horizon, coverage, rng = StableRNG(5),
        inference = WildBootstrap(200; save_draws), kw...
    )
    result = boot()

    @testset "same coverage reproduces the bands" begin
        cb = confidence_bands(result; coverage = [0.95, 0.68, 0.9])
        @test cb.coverage == [0.68, 0.9, 0.95]
        @test all(Array.(cb.lower) .== Array.(result.lower))
        @test all(Array.(cb.upper) .== Array.(result.upper))
        @test cb.irf === result.irf
        @test cb.stderr === result.stderr
        @test cb.bootstrap_draws === result.bootstrap_draws
        @test cb.identification === result.identification
        @test cb.inference === result.inference
    end

    @testset "new coverage matches irf with the same draws" begin
        cov_new = [0.5, 0.8]
        cb = confidence_bands(result; coverage = cov_new)
        direct = boot(; coverage = cov_new)
        @test cb.coverage == cov_new
        @test Array.(cb.lower) == Array.(direct.lower)
        @test Array.(cb.upper) == Array.(direct.upper)
    end

    @testset "rescaled and cumulated results" begin
        scaled = rescale(result, 2.0)
        @test Array.(confidence_bands(scaled; coverage = [0.8]).lower) ≈
            Array.(rescale(confidence_bands(result; coverage = [0.8]), 2.0).lower)
        cum = cumulative_irf(result; vars = [:a])
        cb = confidence_bands(cum; coverage = [0.8])
        direct = boot(; coverage = [0.8], cumulate = [:a])
        @test Array.(cb.lower) ≈ Array.(direct.lower)
        @test Array.(cb.upper) ≈ Array.(direct.upper)
    end

    @testset "errors" begin
        @test_throws "no saved bootstrap draws" confidence_bands(boot(; save_draws = false))
        analytic = irf(model, CholeskyID(); horizon, inference = Analytic())
        @test_throws "no saved bootstrap draws" confidence_bands(analytic)
        @test_throws "coverage levels must be in (0, 1)" confidence_bands(
            result; coverage = [0.9, 1.0]
        )
    end

    @testset "ProxySVARMBB draws" begin
        data_dir = joinpath(@__DIR__, "..", "data")
        Yp = Matrix(CSV.read(joinpath(data_dir, "jl_crossval_Y.csv"), DataFrame))
        proxy = vec(Matrix(CSV.read(joinpath(data_dir, "jl_crossval_proxy.csv"), DataFrame)))
        p = 2
        pmodel = fit(OLSVAR, Yp, p)
        id = IVIdentification(proxy[(p + 1):end], 1)
        mbb(; coverage, kw...) = irf(
            pmodel, id; horizon = 4, coverage, rng = Xoshiro(3),
            inference = ProxySVARMBB(50; save_draws = true, kw...)
        )
        cb = confidence_bands(mbb(; coverage = [0.68, 0.9]); coverage = [0.8])
        direct = mbb(; coverage = [0.8])
        @test Array.(cb.lower) ≈ Array.(direct.lower)
        @test Array.(cb.upper) ≈ Array.(direct.upper)

        ar = mbb(; coverage = [0.9], compute_ar = true, ar_grid = collect(-3.0:0.5:3.0))
        @test_throws "Anderson-Rubin confidence sets" confidence_bands(ar)
    end

    @testset "deprecated forms" begin
        cb = @test_deprecated confidence_bands(result, WildBootstrap; coverage = [0.8])
        @test Array.(cb.lower) == Array.(confidence_bands(result; coverage = [0.8]).lower)

        inf = WildBootstrap(100)
        old = @test_deprecated confidence_bands(
            result, model, CholeskyID(), inf; coverage = [0.8], rng = StableRNG(9)
        )
        new = irf(
            model, CholeskyID(); horizon, inference = inf, coverage = [0.8],
            rng = StableRNG(9)
        )
        @test Array(old.irf) == Array(new.irf)
        @test Array.(old.lower) == Array.(new.lower)
        @test old.inference === inf

        unit = irf(
            model, CholeskyID(); horizon, inference = nothing,
            normalization = UnitEffect()
        )
        old_unit = @test_deprecated confidence_bands(
            unit, model, CholeskyID(), inf; rng = StableRNG(9)
        )
        @test Array(old_unit.irf) ≈ Array(unit.irf)

        @test_throws "rescaled or cumulated" confidence_bands(
            rescale(result, 2.0), model, CholeskyID(), inf
        )
        @test_throws "rescaled or cumulated" confidence_bands(
            cumulative_irf(result; vars = [:a]), model, CholeskyID(), inf
        )
    end
end

@testset "forecast" begin
    Y = post_estimation_data(StableRNG(11), 200, 2)
    p = 2
    model = fit(OLSVAR, Y, p; names = [:a, :b])
    c = model.coefficients.intercept
    A = model.coefficients.lags
    Σ = residual_covariance(model)

    # Conditional mean computed by the VAR recursion.
    function recursion(Y0, h)
        path = [Y0[end - l + 1, :] for l in p:-1:1]
        for _ in 1:h
            push!(path, c + sum(A[:, :, l] * path[end - l + 1] for l in 1:p))
        end
        return permutedims(reduce(hcat, path[(p + 1):end]))
    end

    @testset "point forecast" begin
        for h in (1, 2, 5)
            fc = forecast(model, h)
            @test size(fc) == (h, 2)
            @test fc ≈ recursion(Y, h)
        end
        Y0 = randn(StableRNG(1), 4, 2)
        @test forecast(model, 3; Y_init = Y0) ≈ recursion(Y0, 3)
        # The forecast converges to the unconditional mean.
        μ = ([1.0 0; 0 1] - A[:, :, 1] - A[:, :, 2]) \ c
        @test forecast(model, 400)[end, :] ≈ μ rtol = 1.0e-6
    end

    @testset "draws" begin
        h = 4
        out = forecast(
            model, h; include_draws = true, n_draws = 4000,
            coverage = [0.9, 0.68], rng = StableRNG(3)
        )
        @test out.forecast ≈ recursion(Y, h)
        @test size(out.draws) == (4000, h, 2)
        @test out.coverage == [0.68, 0.9]
        @test size(out.bands.lower) == size(out.bands.upper) == (2, h, 2)
        @test all(out.bands.lower[2, :, :] .<= out.bands.lower[1, :, :])
        @test all(out.bands.lower[1, :, :] .<= out.forecast .<= out.bands.upper[1, :, :])
        @test all(out.bands.upper[1, :, :] .<= out.bands.upper[2, :, :])
        # One step ahead the forecast error is the innovation.
        @test cov(out.draws[:, 1, :]) ≈ Σ rtol = 0.1
        @test vec(mean(out.draws[:, 1, :]; dims = 1)) ≈ out.forecast[1, :] atol = 0.05

        again = forecast(model, h; include_draws = true, n_draws = 4000, rng = StableRNG(3))
        @test again.draws == out.draws
    end

    @testset "errors" begin
        @test_throws "h must be positive" forecast(model, 0)
        @test_throws "n_draws must be positive" forecast(
            model, 2; include_draws = true, n_draws = 0
        )
        @test_throws "coverage levels must be in (0, 1)" forecast(
            model, 2; include_draws = true, coverage = [1.0]
        )
        @test_throws "Y_init must have at least 2 rows" forecast(
            model, 2; Y_init = randn(1, 2)
        )
    end
end

@testset "historical_decomposition" begin
    Y = post_estimation_data(StableRNG(17), 120, 3)
    p = 2
    model = fit(OLSVAR, Y, p; names = [:a, :b, :c])
    n = nobs(model)
    P = rotation_matrix(model, CholeskyID())
    hd = historical_decomposition(model, CholeskyID())

    @test size(hd.contributions) == (n, 3, 3)
    @test size(hd.initial_condition) == (n, 3)
    @test hd.structural_shocks * P' ≈ residuals(model)
    @test hd.initial_condition + dropdims(sum(hd.contributions; dims = 3); dims = 3) ≈
        Y[(p + 1):end, :]
    # The part not explained by shocks is the forecast from the first p observations.
    @test hd.initial_condition ≈ forecast(model, n; Y_init = Y[1:p, :])
    # In the first period only the contemporaneous shock contributes.
    for s in 1:3
        @test hd.contributions[1, :, s] ≈ P[:, s] * hd.structural_shocks[1, s]
    end
end

@testset "VAR(::DataFrame, variables, lags)" begin
    Y = post_estimation_data(StableRNG(23), 100, 3)
    df = DataFrame(a = Y[:, 1], b = Y[:, 2], c = Y[:, 3])
    model = VAR(df, [:c, :a], 2)
    reference = fit(OLSVAR, Y[:, [3, 1]], 2; names = [:c, :a])
    @test model.names == [:c, :a]
    @test coef(model) == coef(reference)
    @test residual_covariance(model) == residual_covariance(reference)

    demeaned = VAR(df, [:a, :b], 1; demean = true)
    @test coef(demeaned) == coef(fit(OLSVAR, Y[:, 1:2], 1; names = [:a, :b], demean = true))

    dfi = DataFrame(x = rand(StableRNG(2), 1:9, 60), y = rand(StableRNG(3), 1:9, 60))
    @test coef(VAR(dfi, [:x, :y], 1)) == coef(fit(OLSVAR, Float64.(Matrix(dfi)), 1))

    @test_throws "Column :d not found in DataFrame" VAR(df, [:a, :d], 1)
end
