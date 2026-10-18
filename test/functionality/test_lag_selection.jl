# ============================================================================
# select_lags: lag-order selection on a common estimation sample
# ============================================================================

using MacroEconometricTools
using Test
using StableRNGs: StableRNG

function lag_selection_data(rng, T)
    A1 = [0.5 0.1; 0.0 0.4]
    A2 = [-0.3 0.0; 0.1 -0.25]
    Y = zeros(T, 2)
    for t in 3:T
        Y[t, :] = A1 * Y[t - 1, :] + A2 * Y[t - 2, :] + randn(rng, 2)
    end
    return Y
end

@testset "select_lags" begin
    Y = lag_selection_data(StableRNG(11), 400)
    max_lags = 6

    @testset "criteria table matches fits on the common sample" begin
        sel = select_lags(Y, max_lags)
        @test sel.table.lags == 1:max_lags
        for p in 1:max_lags
            m = fit(OLSVAR, Y[(max_lags - p + 1):end, :], p)
            @test nobs(m) == size(Y, 1) - max_lags
            @test sel.table.aic[p] ≈ aic(m)
            @test sel.table.bic[p] ≈ bic(m)
            @test sel.table.hqic[p] ≈ hqic(m)
        end
    end

    @testset "chosen order minimizes the requested criterion" begin
        for criterion in (:aic, :bic, :hqic)
            sel = select_lags(Y, max_lags; criterion)
            @test sel.criterion === criterion
            values = getproperty(sel.table, criterion)
            @test values[sel.lags] == minimum(values)
        end
        @test select_lags(Y, max_lags).criterion === :aic
        # The data are a VAR(2) with T = 400; BIC recovers the true order.
        @test select_lags(Y, max_lags; criterion = :bic).lags == 2
    end

    @testset "keywords are passed to every fit" begin
        sel = select_lags(Y, 3; names = [:x, :y], demean = true)
        m = fit(OLSVAR, Y[2:end, :], 2; names = [:x, :y], demean = true)
        @test sel.table.aic[2] ≈ aic(m)
        # The intercept absorbs the sample mean, so demeaning leaves the criteria unchanged.
        @test sel.table.aic ≈ select_lags(Y, 3).table.aic
        c = [BlockExogeneity([:y], [:x])]
        sel_c = select_lags(Y, 3; names = [:x, :y], constraints = c)
        m_c = fit(OLSVAR, Y[3:end, :], 1; names = [:x, :y], constraints = c)
        @test sel_c.table.aic[1] ≈ aic(m_c)
    end

    @testset "errors" begin
        @test_throws "criterion must be :aic, :bic, or :hqic" select_lags(Y, 4; criterion = :fpe)
        @test_throws "max_lags must be positive" select_lags(Y, 0)
        @test_throws "Not enough observations for max_lags = 4" select_lags(Y[1:12, :], 4)
    end
end
