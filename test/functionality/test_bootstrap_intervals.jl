# Bootstrap bands: Efron's percentile or Hall's interval, chosen by `interval` on
# every bootstrap inference type; and the scale of IV impulse responses.
using MacroEconometricTools
using MacroEconometricTools: compute_bands_from_draws, proxy_svar_mbb, iv_impact
using Test
using LinearAlgebra
using Random
using StableRNGs: StableRNG

@testset "Bootstrap intervals: percentile and Hall" begin
    rng = StableRNG(11)
    point = randn(rng, 2, 2, 3)
    draws = randn(rng, 200, 2, 2, 3) .+ 0.3
    lo_p, hi_p = compute_bands_from_draws(point, draws, [0.9])
    lo_h, hi_h = compute_bands_from_draws(point, draws, [0.9]; interval = :hall)
    # Order statistics of the draws, as `_quantile_along_dim1!` takes them.
    sorted = sort(draws[:, 2, 1, 3])
    @test lo_p[1][2, 1, 3] == sorted[10] && hi_p[1][2, 1, 3] == sorted[190]
    # The percentile interval does not depend on the point estimate.
    @test compute_bands_from_draws(point .+ 5, draws, [0.9])[1] == lo_p
    @test lo_h[1] ≈ 2 .* point .- hi_p[1]
    @test hi_h[1] ≈ 2 .* point .- lo_p[1]
    @test_throws ArgumentError compute_bands_from_draws(point, draws, [0.9]; interval = :centered)

    for T in (WildBootstrap, Bootstrap, BlockBootstrap, ProxySVARMBB)
        @test T(10).interval === :percentile
        @test T(10; interval = :hall).interval === :hall
        @test_throws "interval must be one of :percentile, :hall" T(10; interval = :bca)
    end
end

@testset "Bootstrap intervals in irf, confidence_bands and cumulative_irf" begin
    rng = StableRNG(12)
    Y = cumsum(randn(rng, 120, 2); dims = 1) .* 0.1 .+ randn(rng, 120, 2)
    model = fit(OLSVAR, Y, 2)
    for interval in (:percentile, :hall)
        r = irf(model, CholeskyID(); horizon = 6, coverage = [0.68, 0.9],
            inference = WildBootstrap(200; save_draws = true, interval), rng = StableRNG(3))
        lo, hi = compute_bands_from_draws(
            Array(r.irf), Array(r.bootstrap_draws), [0.68, 0.9];
            interval)
        @test Array.(r.lower) == lo && Array.(r.upper) == hi
        c = confidence_bands(r; coverage = [0.8])
        @test Array(only(c.lower)) ==
              only(compute_bands_from_draws(Array(r.irf), Array(r.bootstrap_draws), [0.8];
            interval)[1])
        cum = cumulative_irf(r; vars = [1])
        direct = irf(
            model, CholeskyID(); horizon = 6, coverage = [0.68, 0.9], cumulate = [1],
            inference = WildBootstrap(200; save_draws = true, interval), rng = StableRNG(3))
        @test Array.(cum.lower) ≈ Array.(direct.lower)
    end
end

@testset "IV impact: one-standard-deviation column, P P' = Σ" begin
    rng = StableRNG(13)
    T, K = 300, 3
    e = randn(rng, T, K)
    e[:, 2] .*= 3
    z = e[:, 2] .+ 0.5 .* randn(rng, T)
    Y = zeros(T, K)
    for t in 2:T
        Y[t, :] = 0.5 .* Y[t - 1, :] .+ e[t, :]
    end
    model = fit(OLSVAR, Y, 1)
    id = IVIdentification(z[2:end], 2)
    Σ = Matrix(residual_covariance(model))
    P = rotation_matrix(model, id)
    @test P * P' ≈ Σ
    b = P[:, 2]
    @test dot(b, Σ \ b) ≈ 1
    @test b[2] > 0
    # Unit effect recovers the 2SLS ratios; the one-standard-deviation impact is
    # near the shock's standard deviation, 3.
    β = b ./ b[2]
    @test Array(irf(model, id; horizon = 2, normalization = UnitEffect()).irf)[:, 2, 1] ≈ β
    @test 2.5 < Array(irf(model, id; horizon = 2).irf)[2, 2, 1] < 3.5
    @test first(iv_impact(residuals(model), Σ, reshape(z[2:end], :, 1), 2)) ≈ P

    # MBB draws on the UnitStd scale: the bootstrap's H1 draws, signed so the
    # target's impact is positive, rescaled from divisor T to dof_residual(model).
    r = irf(model, id; horizon = 2, coverage = [0.9],
        inference = ProxySVARMBB(100; save_draws = true), rng = StableRNG(4))
    mbb = proxy_svar_mbb(
        model, id, 2, ProxySVARMBB(100); coverage = [0.9], rng = StableRNG(4))
    factor = sqrt(size(residuals(model), 1) / dof_residual(model))
    expected = mbb.irf_draws .* sign.(mbb.irf_draws[:, 2:2, 1:1]) .* factor
    @test Array(r.bootstrap_draws)[:, :, 2, :] ≈ expected
    @test Array(r.irf)[:, 2, :] ≈ mbb.point_irf .* sign(mbb.point_irf[2, 1]) .* factor
    # The Hall interval of the same draws.
    h = irf(model, id; horizon = 2, coverage = [0.9],
        inference = ProxySVARMBB(100; interval = :hall), rng = StableRNG(4))
    lo, hi = compute_bands_from_draws(Array(r.irf), Array(r.bootstrap_draws), [0.9];
        interval = :hall)
    @test Array(only(h.lower))[:, 2, :] ≈ only(lo)[:, 2, :]
    @test Array(only(h.upper))[:, 2, :] ≈ only(hi)[:, 2, :]

    @test_throws "use normalization = UnitEffect()" irf(model, id; horizon = 2,
        inference = ProxySVARMBB(20; compute_ar = true))
    ar = irf(model, id; horizon = 2, normalization = UnitEffect(), coverage = [0.9],
        inference = ProxySVARMBB(50; compute_ar = true), rng = StableRNG(5))
    @test Array(only(ar.lower))[2, 2, 1] <= 1 <= Array(only(ar.upper))[2, 2, 1]
end
