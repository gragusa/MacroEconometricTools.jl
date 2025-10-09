using BenchmarkTools
using CSV
using DataFrames
using LinearAlgebra
using MacroEconometricTools
using Random
using StableRNGs: StableRNG

const DATA_PATH = joinpath(@__DIR__, "..", "test", "kilian_kim_original_dataset.csv")
const VAR_COLUMN_NAMES = [:DY, :DP, :DPCOM, :FF]

const DEFAULT_SEED = parse(Int, get(ENV, "MET_BENCH_SEED", "20240612"))
const BOOTSTRAP_REPS = parse(Int, get(ENV, "MET_BENCH_REPS", "250"))
const BOOTSTRAP_HORIZON = parse(Int, get(ENV, "MET_BENCH_HORIZON", "20"))
const IRF_BOOTSTRAP_REPS = parse(Int, get(ENV, "MET_BENCH_IRF_REPS", "100"))
const SIGN_MAX_DRAWS = parse(Int, get(ENV, "MET_BENCH_SIGN_DRAWS", "10000"))

"""
    load_sample_matrix()

Reproduce the canonical Kilian-Kim dataset transformations used in tests.
"""
function load_sample_matrix()
    df = DataFrame(CSV.File(DATA_PATH; header=false))
    rename!(df, [:CFNAI, :CPI, :PCOM, :FF])

    dy = df.CFNAI[2:end]
    dp = (log.(df.CPI[2:end]) .- log.(df.CPI[1:(end - 1)])) .* 100
    dpcom = (
        (log.(df.PCOM[2:end]) .- log.(df.CPI[2:end])) .-
        (log.(df.PCOM[1:(end - 1)]) .- log.(df.CPI[1:(end - 1)]))
    ) .* 100
    ff = df.FF[2:end]

    return hcat(dy, dp, dpcom, ff)
end

const SAMPLE_MATRIX = load_sample_matrix()

"""
    build_reference_model()

Estimate a 5-lag OLS VAR on the transformed sample.
"""
function build_reference_model()
    return fit(OLSVAR, SAMPLE_MATRIX, 5; names=VAR_COLUMN_NAMES)
end

const REFERENCE_MODEL = build_reference_model()

function identity_matrix_int(n::Int)
    M = zeros(Int, n, n)
    @inbounds for i in 1:n
        M[i, i] = 1
    end
    return M
end

const CHOLESKY_ID = CholeskyID()
const SIGN_RESTRICTION = SignRestriction(identity_matrix_int(length(VAR_COLUMN_NAMES)), 0)

"""
    _init_distributed()

Optionally prepare Distributed for parallel benchmarks.
"""
function _init_distributed()
    if get(ENV, "MET_BENCH_ENABLE_DISTRIBUTED", "false") != "true"
        return false
    end

    try
        dist = Base.require(Main, :Distributed)
        desired = parse(Int, get(ENV, "MET_BENCH_NWORKERS", string(max(2, Sys.CPU_THREADS))))
        desired = max(desired, 2)

        if dist.nworkers() < desired
            dist.addprocs(desired - dist.nworkers())
        end

        Core.eval(Main, :(Distributed.@everywhere using MacroEconometricTools))

        return dist.nworkers() > 1
    catch err
        @warn "Unable to initialise distributed benchmarking; skipping distributed benchmarks" exception=err
        return false
    end
end

const DISTRIBUTED_READY = _init_distributed()

const SUITE = BenchmarkGroup()

SUITE["estimation"] = BenchmarkGroup()
SUITE["estimation"]["ols_var_5lags"] =
    @benchmarkable fit(OLSVAR, $SAMPLE_MATRIX, 5; names=$VAR_COLUMN_NAMES)

SUITE["irf"] = BenchmarkGroup()
SUITE["irf"]["point_response"] =
    @benchmarkable irf($REFERENCE_MODEL, $CHOLESKY_ID; horizon=$BOOTSTRAP_HORIZON, inference=:none)

SUITE["irf"]["bootstrap_response"] =
    @benchmarkable begin
        irf($REFERENCE_MODEL, $CHOLESKY_ID;
            horizon=$BOOTSTRAP_HORIZON,
            inference=:bootstrap,
            bootstrap_reps=$IRF_BOOTSTRAP_REPS,
            bootstrap_method=:wild,
            rng=StableRNG($DEFAULT_SEED))
    end

SUITE["bootstrap"] = BenchmarkGroup()
SUITE["bootstrap"]["serial_wild"] =
    @benchmarkable begin
        bootstrap_irf($REFERENCE_MODEL, $CHOLESKY_ID,
                      $BOOTSTRAP_HORIZON, $BOOTSTRAP_REPS;
                      method=:wild, parallel=:none,
                      rng=StableRNG($DEFAULT_SEED))
    end

if DISTRIBUTED_READY
    SUITE["bootstrap"]["distributed_wild"] =
        @benchmarkable begin
            bootstrap_irf($REFERENCE_MODEL, $CHOLESKY_ID,
                          $BOOTSTRAP_HORIZON, $BOOTSTRAP_REPS;
                          method=:wild, parallel=:distributed,
                          rng=StableRNG($DEFAULT_SEED))
        end
end

SUITE["sign_restrictions"] = BenchmarkGroup()
SUITE["sign_restrictions"]["serial"] =
    @benchmarkable begin
        MacroEconometricTools.identify_sign($REFERENCE_MODEL, $SIGN_RESTRICTION;
                      max_draws=$SIGN_MAX_DRAWS,
                      parallel=:none,
                      verbose=false,
                      rng=StableRNG($DEFAULT_SEED))
    end

if DISTRIBUTED_READY
    SUITE["sign_restrictions"]["distributed"] =
        @benchmarkable begin
            MacroEconometricTools.identify_sign($REFERENCE_MODEL, $SIGN_RESTRICTION;
                          max_draws=$SIGN_MAX_DRAWS,
                          parallel=:distributed,
                          verbose=false,
                          rng=StableRNG($DEFAULT_SEED))
        end
end


