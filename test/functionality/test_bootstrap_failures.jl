using Test
using MacroEconometricTools
using MacroEconometricTools: bootstrap_irf
using LinearAlgebra
using Statistics: std
using StableRNGs

const MET = MacroEconometricTools

# Cholesky identification that makes chosen calls of `rotation_matrix` fail.
# `failures` maps `k` to what call `k + 1` does: throw the given exception, or return
# a rotation filled with `NaN`. Under `irf`, call 1 is the point estimate, so key `r`
# is bootstrap replication `r`.
struct FlakyCholesky <: MET.AbstractIdentification
    calls::Base.RefValue{Int}
    failures::Dict{Int, Any}
end
FlakyCholesky(failures) = FlakyCholesky(Ref(0), Dict{Int, Any}(failures))

function MET.rotation_matrix(model::MET.VARModel, id::FlakyCholesky)
    id.calls[] += 1
    P = Matrix(cholesky(model.Σ).L)
    action = get(id.failures, id.calls[] - 1, nothing)
    action === nothing && return P
    action === NaN && return fill!(P, NaN)
    throw(action)
end

@testset "Bootstrap failure policy" begin
    rng = StableRNG(11)
    Y = randn(rng, 150, 2)
    for t in 2:150
        Y[t, :] .+= 0.4 .* Y[t - 1, :]
    end
    model = fit(OLSVAR, Y, 1; names = [:a, :b])
    reps = 40
    failures = Dict(
        3 => PosDefException(1), 9 => SingularException(2),
        17 => LAPACKException(1), 25 => DomainError(-1.0), 31 => NaN,
    )

    @testset "Failure classification" begin
        out = zeros(2)
        for err in (PosDefException(1), SingularException(1), LAPACKException(1), DomainError(-1))
            @test !MET._try_draw!(o -> throw(err), out)
        end
        @test !MET._try_draw!(o -> fill!(o, Inf), out)
        @test MET._try_draw!(o -> fill!(o, 1.0), out)
        @test_throws "not a draw failure" MET._try_draw!(o -> error("not a draw failure"), out)
        @test_throws BoundsError MET._try_draw!(o -> o[3], out)
    end

    for (name, inf) in (
            ("Wild bootstrap", WildBootstrap(reps; save_draws = true, max_failure_rate = 0.2)),
            ("Standard bootstrap", Bootstrap(reps; save_draws = true, max_failure_rate = 0.2)),
            (
                "Block bootstrap",
                BlockBootstrap(reps; block_length = 5, save_draws = true, max_failure_rate = 0.2),
            ),
        )
        @testset "$name" begin
            id = FlakyCholesky(failures)
            result = @test_logs (:warn, r"5 / 40 replications failed") irf(
                model, id; horizon = 6, inference = inf, rng = StableRNG(5)
            )
            draws = result.bootstrap_draws
            @test size(draws, 1) == reps - length(failures)
            @test all(isfinite, draws)
            @test all(isfinite, result.stderr)
            @test all(b -> all(isfinite, b), result.lower)
            # Bands and standard errors are those of the successful draws alone.
            point = Array(result.irf)
            lo, hi = MET.compute_bands_from_draws(point, Array(draws), result.coverage)
            @test all(Array.(result.lower) .== lo)
            @test all(Array.(result.upper) .== hi)
            @test Array(result.stderr) ≈ dropdims(std(Array(draws); dims = 1); dims = 1)
        end
    end

    @testset "Failure rate above max_failure_rate throws" begin
        @test_throws "more than max_failure_rate = 0.1" irf(
            model, FlakyCholesky(failures); horizon = 4,
            inference = WildBootstrap(reps), rng = StableRNG(5)
        )
        all_fail = Dict(r => PosDefException(1) for r in 0:5)
        @test_throws "5 / 5 replications failed" bootstrap_irf(
            model, FlakyCholesky(all_fail), WildBootstrap(5; max_failure_rate = 0.99);
            horizon = 4, rng = StableRNG(5)
        )
    end

    @testset "Unexpected exceptions propagate" begin
        id = FlakyCholesky(Dict(2 => ArgumentError("unexpected in replication")))
        @test_throws "unexpected in replication" irf(
            model, id; horizon = 4, inference = Bootstrap(5), rng = StableRNG(5)
        )
    end

    @testset "max_failure_rate validation" begin
        for T in (WildBootstrap, Bootstrap, BlockBootstrap, ProxySVARMBB)
            @test_throws "max_failure_rate must lie in [0, 1)" T(10; max_failure_rate = 1)
            @test_throws "max_failure_rate must lie in [0, 1)" T(10; max_failure_rate = -0.1)
            @test T(10; max_failure_rate = 0).max_failure_rate == 0
        end
    end

    @testset "Analytic() requires the default Cholesky identification" begin
        @test_throws "Analytic() inference is valid only for CholeskyID()" irf(
            model, CholeskyID([2, 1]); horizon = 4, inference = Analytic()
        )
        @test_throws "got FlakyCholesky" irf(
            model, FlakyCholesky(Dict()); horizon = 4, inference = Analytic()
        )
        z = model.residuals[:, 1] .+ randn(StableRNG(3), size(model.residuals, 1))
        @test_throws "got IVIdentification" irf(
            model, IVIdentification(z, 1); horizon = 4, inference = Analytic()
        )
    end
end
