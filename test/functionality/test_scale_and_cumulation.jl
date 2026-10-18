# ============================================================================
# rescale, get_scale, cumulative_irf, and the `scale`/`cumulate` keywords of irf
# ============================================================================

using MacroEconometricTools
using Test
using Random
using StableRNGs: StableRNG

@testset "Scale and cumulation" begin
    rng = StableRNG(2024)
    T, n_v, n_l = 150, 3, 2
    Y = randn(rng, T, n_v)
    for t in 3:T
        Y[t, :] = 0.5 * Y[t - 1, :] + 0.2 * Y[t - 2, :] + 0.3 * randn(rng, n_v)
    end
    model = fit(OLSVAR, Y, n_l; names = [:a, :b, :c])
    horizon = 8
    boot(; kw...) = irf(
        model, CholeskyID(); horizon, rng = StableRNG(5),
        inference = WildBootstrap(200; save_draws = true), kw...
    )
    result = boot()

    @testset "rescale with pairs" begin
        scaled = rescale(result, :a => -2.0, :c => 100)
        f = [-2.0, 1.0, 100.0]
        @test typeof(scaled) == typeof(result)
        for j in 1:n_v
            @test Array(scaled.irf)[:, j, :] ≈ f[j] .* Array(result.irf)[:, j, :]
            @test Array(scaled.stderr)[:, j, :] ≈ abs(f[j]) .* Array(result.stderr)[:, j, :]
            @test Array(get_draws(scaled))[:, :, j, :] ≈
                f[j] .* Array(get_draws(result))[:, :, j, :]
        end
        # A negative factor swaps lower and upper.
        for k in eachindex(result.coverage)
            @test Array(scaled.lower[k])[:, 1, :] ≈ -2.0 .* Array(result.upper[k])[:, 1, :]
            @test Array(scaled.upper[k])[:, 1, :] ≈ -2.0 .* Array(result.lower[k])[:, 1, :]
            @test Array(scaled.lower[k])[:, 3, :] ≈ 100 .* Array(result.lower[k])[:, 3, :]
            @test all(Array(scaled.lower[k]) .<= Array(scaled.upper[k]))
        end
        # Input unchanged
        @test Array(result.irf) == Array(boot().irf)
        @test get_scale(result).scale == ones(n_v)
    end

    @testset "rescale agrees with bands recomputed from scaled draws" begin
        scaled = rescale(result, :a => 2.0, :c => 100)
        lo, hi = MacroEconometricTools.compute_bands_from_draws(
            Array(scaled.irf), Array(get_draws(scaled)), scaled.coverage
        )
        for k in eachindex(scaled.coverage)
            @test lo[k] ≈ Array(scaled.lower[k])
            @test hi[k] ≈ Array(scaled.upper[k])
        end
    end

    @testset "rescale with a scalar" begin
        @test Array(rescale(result, 3).irf) ≈ 3 .* Array(result.irf)
        @test get_scale(rescale(result, 3)).scale == fill(3.0, n_v)
    end

    @testset "rescale without draws" begin
        r = irf(model, CholeskyID(); horizon, inference = Analytic())
        scaled = @test_logs rescale(r, :b => -1)
        @test Array(scaled.lower[1])[:, 2, :] ≈ -Array(r.upper[1])[:, 2, :]
        @test !has_draws(scaled)
    end

    @testset "get_scale composes successive rescale calls" begin
        s = get_scale(rescale(rescale(result, :a => -1), :a => 4, :b => 0.5))
        @test s isa IRFScale
        @test s.scale == [-4.0, 0.5, 1.0]
        @test s.names == [:a, :b, :c]
        @test s.normalization == UnitStd()
        @test s.impact_diagonal == get_scale(result).impact_diagonal
    end

    @testset "rescale errors" begin
        @test_throws "Shock :z not found" rescale(result, :z => 2)
        @test_throws "scale factors must be finite" rescale(result, :a => Inf)
        @test_throws "scale factors must be finite" rescale(result, NaN)
    end

    @testset "deprecated irf(...; scale) equals rescale" begin
        kw = @test_deprecated boot(scale = -0.25)
        ph = rescale(result, -0.25)
        @test Array(kw.irf) ≈ Array(ph.irf)
        @test Array(kw.stderr) ≈ Array(ph.stderr)
        @test Array(get_draws(kw)) ≈ Array(get_draws(ph))
        @test all(Array(kw.lower[k]) ≈ Array(ph.lower[k]) for k in eachindex(kw.lower))
        @test all(Array(kw.upper[k]) ≈ Array(ph.upper[k]) for k in eachindex(kw.upper))
        @test get_scale(kw).scale == get_scale(ph).scale
    end

    @testset "rescale(::SignRestrictedIRFResult)" begin
        R = zeros(Int, n_v, n_v)
        R[1, 1] = 1
        sr = irf(model, SignRestriction(R, 0); horizon, n_draws = 40, rng = StableRNG(3))
        scaled = rescale(sr, :a => -3)
        @test Array(scaled.irf_draws)[:, :, 1, :] ≈ -3 .* Array(sr.irf_draws)[:, :, 1, :]
        @test Array(scaled.irf_median)[:, 1, :] ≈ -3 .* Array(sr.irf_median)[:, 1, :]
        @test Array(scaled.lower[1])[:, 1, :] ≈ -3 .* Array(sr.upper[1])[:, 1, :]
        @test get_scale(scaled).scale == [-3.0, 1.0, 1.0]
        kw = @test_deprecated irf(
            model, SignRestriction(R, 0); horizon, n_draws = 40, rng = StableRNG(3),
            scale = 2
        )
        @test Array(kw.irf_median) ≈ Array(rescale(sr, 2).irf_median)
    end

    @testset "cumulative_irf agrees with irf(...; cumulate)" begin
        kw = boot(cumulate = [:b])
        ph = cumulative_irf(result; vars = [:b])
        @test Array(kw.irf) == Array(ph.irf)
        @test Array(get_draws(kw)) == Array(get_draws(ph))
        @test Array(kw.stderr) ≈ Array(ph.stderr)
        @test all(Array(kw.lower[k]) == Array(ph.lower[k]) for k in eachindex(kw.lower))
        @test all(Array(kw.upper[k]) == Array(ph.upper[k]) for k in eachindex(kw.upper))
        @test kw.metadata.cumulated_vars == ph.metadata.cumulated_vars == [:b]
    end

    @testset "cumulative_irf metadata and repeated cumulation" begin
        once = cumulative_irf(result; vars = [2])
        twice = cumulative_irf(once; vars = [:c])
        @test twice.metadata.cumulated_vars == [:b, :c]
        @test Array(twice.irf) == Array(cumulative_irf(result; vars = [:b, :c]).irf)
        @test_throws "already cumulated" cumulative_irf(once; vars = [:b, :c])
        @test_throws "already cumulated" cumulative_irf(boot(cumulate = [:a]))
        @test cumulative_irf(result).metadata.cumulated_vars == [:a, :b, :c]
    end

    @testset "cumulative_irf requires draws" begin
        for inference in (nothing, Analytic(), WildBootstrap(30))
            r = irf(model, CholeskyID(); horizon, inference, rng = StableRNG(1))
            @test_throws "requires stored bootstrap draws" cumulative_irf(r)
        end
    end

    @testset "Anderson-Rubin bands" begin
        rng = StableRNG(77)
        T_iv = 200
        ε = randn(rng, T_iv, n_v)
        B0 = [1.0 0.0 0.0; 0.5 1.0 0.0; 0.3 0.2 1.0]
        Yiv = zeros(T_iv, n_v)
        for t in 2:T_iv
            Yiv[t, :] = 0.5 * Yiv[t - 1, :] + B0 * ε[t, :]
        end
        m = fit(OLSVAR, Yiv, 1; names = [:a, :b, :c])
        z = ε[2:end, 1] .+ 0.3 .* randn(rng, T_iv - 1)
        id = IVIdentification(z, 1)
        inf = ProxySVARMBB(50; block_length = 4, compute_ar = true, save_draws = true)
        r = irf(m, id; horizon = 4, inference = inf, coverage = [0.9], rng = StableRNG(1))

        scaled = rescale(r, :a => -2)
        @test Array(scaled.lower[1])[:, 1, :] ≈ -2 .* Array(r.upper[1])[:, 1, :]
        @test Array(scaled.upper[1])[:, 1, :] ≈ -2 .* Array(r.lower[1])[:, 1, :]

        @test_throws "Anderson-Rubin" cumulative_irf(r)
        @test_throws "Anderson-Rubin" irf(
            m, id; horizon = 4, inference = inf, cumulate = [:b], rng = StableRNG(1)
        )
    end
end
