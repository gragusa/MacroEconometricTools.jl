# Coverage-driven results of `proxy_svar_mbb` and `msw_confidence_set`.

using MacroEconometricTools
using MacroEconometricTools: proxy_svar_mbb, msw_confidence_set
using Test
using Random
using CSV
using DataFrames

@testset "Proxy-SVAR inference at requested coverage levels" begin
    data_dir = joinpath(@__DIR__, "..", "data")
    Y = Matrix(CSV.read(joinpath(data_dir, "jl_crossval_Y.csv"), DataFrame))
    proxy = vec(Matrix(CSV.read(joinpath(data_dir, "jl_crossval_proxy.csv"), DataFrame)))
    p = 2
    K = 2
    horizon = 6
    n_imp = horizon + 1
    model = fit(OLSVAR, Y, p)
    id = IVIdentification(proxy[(p + 1):end], 1)
    grid = collect(range(-3.0, 3.0; length = 61))
    inf = ProxySVARMBB(150; block_length = 4, compute_ar = true, ar_grid = grid)

    @testset "public, not exported" begin
        for name in (:proxy_svar_mbb, :msw_confidence_set)
            @test Base.ispublic(MacroEconometricTools, name)
            @test !Base.isexported(MacroEconometricTools, name)
        end
    end

    @testset "proxy_svar_mbb" begin
        cov = [0.95, 0.5, 0.8]
        mbb = proxy_svar_mbb(model, id, horizon, inf; coverage = cov, rng = Xoshiro(7))
        n = size(mbb.irf_norm_draws, 1)

        @test mbb.coverage == [0.5, 0.8, 0.95]
        @test size(mbb.irf_draws) == size(mbb.irf_norm_draws) == (n, K, n_imp)
        @test size(mbb.point_irf_norm) == (K, n_imp)
        for bands in (mbb.irf_bands, mbb.irf_norm_bands, mbb.fevd_bands, mbb.halls_irf_norm_bands)
            @test length(bands.lower) == length(bands.upper) == 3
            @test all(m -> size(m) == (K, n_imp), bands.lower)
        end

        # Percentile bounds are order statistics of the draws: rank round(q * 150) + 1
        # for q = 0.25/0.75, 0.10/0.90, 0.025/0.975 (`round` rounds ties to even).
        @test n == 150
        sorted = sort(mbb.irf_norm_draws; dims = 1)
        for (i, (rank_lo, rank_hi)) in enumerate(((39, 113), (16, 136), (5, 147)))
            @test mbb.irf_norm_bands.lower[i] == sorted[rank_lo, :, :]
            @test mbb.irf_norm_bands.upper[i] == sorted[rank_hi, :, :]
            @test mbb.halls_irf_norm_bands.lower[i] ≈ 2 .* mbb.point_irf_norm .- sorted[rank_hi, :, :]
            @test mbb.halls_irf_norm_bands.upper[i] ≈ 2 .* mbb.point_irf_norm .- sorted[rank_lo, :, :]
        end

        # AR sets: inclusion follows the rates, nested in the level.
        @test size(mbb.ar.rates) == (length(grid), K, n_imp)
        inc80 = 0.1 .<= mbb.ar.rates .<= 0.9
        inc80[findall(≈(-1.0), grid), 1, 1] .= true
        @test mbb.ar.included[2] == inc80
        @test all(mbb.ar.included[1] .<= mbb.ar.included[2] .<= mbb.ar.included[3])

        # A level's results do not depend on the other requested levels.
        mbb80 = proxy_svar_mbb(model, id, horizon, inf; coverage = [0.8], rng = Xoshiro(7))
        @test mbb80.irf_norm_bands.lower[1] == mbb.irf_norm_bands.lower[2]
        @test mbb80.ar.included[1] == mbb.ar.included[2]

        @test_throws "coverage levels must be in (0, 1)" proxy_svar_mbb(
            model, id, horizon, inf; coverage = [0.9, 1.0]
        )
    end

    @testset "irf uses the AR sets at the requested level" begin
        r = irf(
            model, id; horizon, inference = inf, coverage = [0.8], rng = Xoshiro(11)
        )
        mbb = proxy_svar_mbb(model, id, horizon, inf; coverage = [0.8], rng = Xoshiro(11))
        norm_factor = Array(r.irf)[1, 1, 1] / inf.norm_scale
        lower = Array(r.lower[1])
        upper = Array(r.upper[1])
        for k in 1:K, h in 1:n_imp
            g = grid[mbb.ar.included[1][:, k, h]] .* norm_factor
            @test lower[k, 1, h] ≈ minimum(g)
            @test upper[k, 1, h] ≈ maximum(g)
        end
    end

    @testset "msw_confidence_set" begin
        # χ²(1) quantiles at 0.5, 0.68, 0.9, 0.95, 0.99
        crit = [
            0.45493642311957266, 0.9889464814780231, 2.705543454095415,
            3.8414588206941245, 6.634896601021214,
        ]
        cov = [0.99, 0.5, 0.68, 0.9, 0.95]
        msw = msw_confidence_set(model, id; coverage = cov, horizon)
        @test msw.coverage == sort(cov)
        @test msw.bounded == (msw.wald_stat .> crit)
        @test all(cs -> size(cs) == (2, K, n_imp), msw.cs_irf_norm)
        for i in 1:4, k in 1:K, h in 2:n_imp
            @test msw.cs_irf_norm[i][1, k, h] >= msw.cs_irf_norm[i + 1][1, k, h]
            @test msw.cs_irf_norm[i][2, k, h] <= msw.cs_irf_norm[i + 1][2, k, h]
        end
        msw68 = msw_confidence_set(model, id; coverage = [0.68], horizon)
        @test msw68.cs_irf_norm[1] == msw.cs_irf_norm[2]

        # A weak instrument gives unbounded sets at high coverage.
        weak = proxy[(p + 1):end] .+ 20 .* randn(Xoshiro(3), length(proxy) - p)
        msw_weak = msw_confidence_set(model, IVIdentification(weak, 1); horizon)
        crit3 = crit[[2, 3, 4]]
        @test msw_weak.bounded == (msw_weak.wald_stat .> crit3)
        @test !all(msw_weak.bounded)
        for (cs, bnd) in zip(msw_weak.cs_irf_norm, msw_weak.bounded)
            @test size(cs, 1) == (bnd ? 2 : 4)
        end

        @test_throws "coverage levels must be in (0, 1)" msw_confidence_set(
            model, id; coverage = [0.0, 0.9], horizon
        )
    end
end
