# ============================================================================
# Tests for the new SVAR-IV API: IVIdentification(Z, target)
# ============================================================================
# Verifies that the new API (instrument in identification, not in model)
# produces identical results to the old API and works with all inference types.

using MacroEconometricTools
using Test
using LinearAlgebra
using Random
using StableRNGs: StableRNG

# ============================================================================
# Helper: simulate a bivariate VAR with known structural parameters
# ============================================================================
function simulate_var_dgp(T::Int, A_coefs::Vector{Matrix{Float64}},
        H::Matrix{Float64}; rng = StableRNG(42), burn::Int = 500)
    p = length(A_coefs)
    K = size(H, 1)
    T_total = T + burn

    ε = randn(rng, T_total, K)
    Y = zeros(T_total, K)

    for t in (p + 1):T_total
        for lag in 1:p
            Y[t, :] .+= A_coefs[lag] * Y[t - lag, :]
        end
        Y[t, :] .+= H * ε[t, :]
    end

    return Y[(burn + 1):end, :], ε[(burn + 1):end, :]
end

# ============================================================================
# Setup
# ============================================================================
const K = 3
const T_sim = 500
const H_true = [1.0 0.0 0.0;
                0.5 0.8 0.0;
                0.3 0.2 1.0]
const A1 = [0.3 0.0 0.1;
            0.1 0.5 0.0;
            0.0 0.0 0.4]
const p_lag = 1

Y, ε = simulate_var_dgp(T_sim, [A1], H_true; rng = StableRNG(2024))

# ============================================================================
# Test 1: New API matches old API exactly
# ============================================================================
@testset "New API matches old API" begin
    # Old API
    Z_trimmed = reshape(ε[(p_lag + 1):end, 1], :, 1)
    model_old = fit(IVSVAR, Y, p_lag;
        instrument = ExternalInstrument(Z_trimmed, 1), names = [:Y1, :Y2, :Y3])
    irf_old = irf(model_old, IVIdentification(); horizon = 10)

    # New API
    model_new = fit(OLSVAR, Y, p_lag; names = [:Y1, :Y2, :Y3])
    id = IVIdentification(ε[:, 1], 1)  # full-length, auto-trimmed
    irf_new = irf(model_new, id; horizon = 10)

    @test Array(irf_old.irf) ≈ Array(irf_new.irf) atol=1e-10
end

# ============================================================================
# Test 2: rotation_matrix does 2SLS
# ============================================================================
@testset "rotation_matrix with IVIdentification(Z, target)" begin
    model = fit(OLSVAR, Y, p_lag; names = [:Y1, :Y2, :Y3])
    id = IVIdentification(ε[:, 1], 1)

    P = rotation_matrix(model, id)
    @test size(P) == (K, K)
    @test P[1, 1] ≈ 1.0 atol=1e-10  # unit effect normalization
end

# ============================================================================
# Test 3: Diagnostics with (model, id)
# ============================================================================
@testset "Diagnostics: first_stage_F and iv_summary" begin
    model = fit(OLSVAR, Y, p_lag; names = [:Y1, :Y2, :Y3])
    id = IVIdentification(ε[:, 1], 1)

    F = first_stage_F(model, id)
    @test F > 1000  # perfect instrument

    # Compare with old API
    model_old = fit(IVSVAR, Y, p_lag;
        instrument = ExternalInstrument(ε[:, 1], 1), names = [:Y1, :Y2, :Y3])
    F_old = first_stage_F(model_old)
    @test F ≈ F_old atol=1e-10

    # iv_summary should not error
    old_stdout = stdout
    rd, wr = redirect_stdout()
    iv_summary(model, id)
    redirect_stdout(old_stdout)
    close(wr)
    output = read(rd, String)
    @test contains(output, "SVAR-IV")
    @test contains(output, "Y1")
end

# ============================================================================
# Test 4: irf with various inference types
# ============================================================================
@testset "irf with WildBootstrap + IVIdentification" begin
    model = fit(OLSVAR, Y, p_lag; names = [:Y1, :Y2, :Y3])
    id = IVIdentification(ε[:, 1], 1)

    result = irf(model, id; horizon = 10,
        inference = WildBootstrap(50; save_draws = false),
        rng = Random.Xoshiro(42))

    @test size(result.irf) == (K, K, 11)
    @test length(result.lower) == 3  # 68%, 90%, 95%
    @test all(isfinite, Array(result.irf))
end

@testset "irf with Bootstrap + IVIdentification" begin
    model = fit(OLSVAR, Y, p_lag; names = [:Y1, :Y2, :Y3])
    id = IVIdentification(ε[:, 1], 1)

    result = irf(model, id; horizon = 10,
        inference = Bootstrap(50),
        rng = Random.Xoshiro(42))

    @test size(result.irf) == (K, K, 11)
    @test length(result.lower) == 3
end

@testset "irf with BlockBootstrap + IVIdentification" begin
    model = fit(OLSVAR, Y, p_lag; names = [:Y1, :Y2, :Y3])
    id = IVIdentification(ε[:, 1], 1)

    result = irf(model, id; horizon = 10,
        inference = BlockBootstrap(50; block_length = 5),
        rng = Random.Xoshiro(42))

    @test size(result.irf) == (K, K, 11)
    @test length(result.lower) == 3
end

@testset "irf with ProxySVARMBB + IVIdentification" begin
    model = fit(OLSVAR, Y, p_lag; names = [:Y1, :Y2, :Y3])
    id = IVIdentification(ε[:, 1], 1)

    result = irf(model, id; horizon = 10,
        inference = ProxySVARMBB(50; block_length = 4),
        rng = Random.Xoshiro(42))

    @test size(result.irf) == (K, K, 11)
    @test length(result.lower) == 3
end

# ============================================================================
# Test 5: ProxySVARMBB errors with non-IV identification
# ============================================================================
@testset "ProxySVARMBB errors with CholeskyID" begin
    model = fit(OLSVAR, Y, p_lag; names = [:Y1, :Y2, :Y3])

    @test_throws ArgumentError irf(model, CholeskyID(); horizon = 10,
        inference = ProxySVARMBB(50))
end

# ============================================================================
# Test 6: proxy_svar_mbb with (model, id) — direct call
# ============================================================================
@testset "proxy_svar_mbb(model, id, ...)" begin
    model = fit(OLSVAR, Y, p_lag; names = [:Y1, :Y2, :Y3])
    id = IVIdentification(ε[:, 1], 1)

    mbb = proxy_svar_mbb(model, id, 10,
        ProxySVARMBB(50; block_length = 4, norm_scale = -1.0);
        rng = Random.Xoshiro(42))

    @test all(isfinite, mbb.ci68_irf_norm)
    @test all(isfinite, mbb.ci95_irf_norm)
    @test mbb.point_irf_norm[1, 1] ≈ -1.0 atol=1e-10
end

# ============================================================================
# Test 7: AR confidence sets with default grid
# ============================================================================
@testset "AR with default grid" begin
    model = fit(OLSVAR, Y, p_lag; names = [:Y1, :Y2, :Y3])
    id = IVIdentification(ε[:, 1], 1)

    # ar_grid=nothing should use default [-10, 10] x 201
    mbb = proxy_svar_mbb(model, id, 5,
        ProxySVARMBB(50; block_length = 4, compute_ar = true);
        rng = Random.Xoshiro(42))

    @test mbb.ar !== nothing
    @test length(mbb.ar.grid) == 201
    @test mbb.ar.grid[1] ≈ -10.0
    @test mbb.ar.grid[end] ≈ 10.0
end

@testset "AR sets nested: 68% ⊂ 90% ⊂ 95%" begin
    model = fit(OLSVAR, Y, p_lag; names = [:Y1, :Y2, :Y3])
    id = IVIdentification(ε[:, 1], 1)

    mbb = proxy_svar_mbb(model, id, 5,
        ProxySVARMBB(100; block_length = 4, compute_ar = true);
        rng = Random.Xoshiro(42))

    for k in 1:K, h in 1:6

        @test sum(mbb.ar.index68[:, h, k]) ≤ sum(mbb.ar.index90[:, h, k])
        @test sum(mbb.ar.index90[:, h, k]) ≤ sum(mbb.ar.index95[:, h, k])
    end
end

# ============================================================================
# Test 8: AR convex hull in irf() bands
# ============================================================================
@testset "irf with compute_ar=true produces plottable bands" begin
    model = fit(OLSVAR, Y, p_lag; names = [:Y1, :Y2, :Y3])
    id = IVIdentification(ε[:, 1], 1)

    result = irf(model, id; horizon = 5,
        inference = ProxySVARMBB(100; block_length = 4, compute_ar = true),
        coverage = [0.68, 0.95],
        rng = Random.Xoshiro(42))

    # Bands should exist and 68% ⊂ 95%
    lb1 = Array(result.lower[1])  # (variable, shock, horizon)
    ub1 = Array(result.upper[1])
    lb2 = Array(result.lower[2])
    ub2 = Array(result.upper[2])
    for h in 1:6
        w68 = ub1[1, 1, h] - lb1[1, 1, h]
        w95 = ub2[1, 1, h] - lb2[1, 1, h]
        if isfinite(w68) && isfinite(w95)
            @test w68 ≤ w95 + 1e-10
        end
    end
end

# ============================================================================
# Test 9: msw_confidence_set with (model, id)
# ============================================================================
@testset "msw_confidence_set(model, id)" begin
    model = fit(OLSVAR, Y, p_lag; names = [:Y1, :Y2, :Y3])
    id = IVIdentification(ε[:, 1], 1)

    msw = msw_confidence_set(model, id; horizon = 10)

    @test msw.wald_stat > 0
    @test msw.bounded68  # strong instrument → bounded

    # Compare with old API
    model_old = fit(IVSVAR, Y, p_lag;
        instrument = ExternalInstrument(ε[:, 1], 1), names = [:Y1, :Y2, :Y3])
    msw_old = msw_confidence_set(model_old; horizon = 10)

    @test msw.wald_stat ≈ msw_old.wald_stat atol=1e-10
    @test msw.cs68_irf_norm ≈ msw_old.cs68_irf_norm atol=1e-10
end

# ============================================================================
# Test 10: IVIdentification constructors
# ============================================================================
@testset "IVIdentification constructors" begin
    # Vector
    id1 = IVIdentification(randn(100), 1)
    @test id1.instrument isa ExternalInstrument
    @test id1.instrument.target_shock == 1

    # Matrix
    id2 = IVIdentification(randn(100, 2), 1; method = :liml)
    @test id2.instrument.method == :liml

    # ExternalInstrument
    inst = ExternalInstrument(randn(100), 2)
    id3 = IVIdentification(inst)
    @test id3.instrument.target_shock == 2

    # Empty (backward compat)
    id4 = IVIdentification()
    @test id4.instrument === nothing
end

# ============================================================================
# Permutation invariance: target_shock ≠ 1 must yield results equivalent to
# target_shock = 1 on reordered data. This catches any remaining hardcoded
# position-1 assumptions in the identification, bootstrap, or MSW code.
# ============================================================================

# Reorder Y so the instrumented variable (originally column 1) lives at
# column `target`. Returns `perm` s.t. `Y_perm = Y[:, perm]`, i.e.
# `Y_perm[:, new] = Y[:, perm[new]]`. `perm[target] = 1` puts the instrumented
# var at position `target`; remaining old columns (2..K) fill the other slots.
function _permute_cols(K::Int, target::Int)
    other_olds = [k for k in 2:K]   # old columns that are NOT the instrumented one
    perm = zeros(Int, K)
    perm[target] = 1
    fill_pos = 1
    for old in other_olds
        while fill_pos == target
            fill_pos += 1
        end
        perm[fill_pos] = old
        fill_pos += 1
    end
    return perm
end

@testset "Permutation invariance: identified IRF column" begin
    # IRF axes: (variable, shock, horizon). Y_perm = Y[:, perm], so
    # r_perm.irf[invperm(perm), target, :] reorders permuted variables back
    # to original positions for comparison against r_base.irf[:, 1, :].
    names_base = [:Y1, :Y2, :Y3]
    model_base = fit(OLSVAR, Y, p_lag; names = names_base)
    id_base = IVIdentification(ε[:, 1], 1)
    r_base = irf(model_base, id_base; horizon = 10)

    for target in 2:K
        perm = _permute_cols(K, target)
        Y_perm = Y[:, perm]
        names_perm = names_base[perm]
        model_perm = fit(OLSVAR, Y_perm, p_lag; names = names_perm)

        id_perm = IVIdentification(ε[:, 1], target)
        r_perm = irf(model_perm, id_perm; horizon = 10)

        base_id_col = Array(r_base.irf)[:, 1, :]                   # (K, n_imp)
        perm_id_col = Array(r_perm.irf)[invperm(perm), target, :]  # (K, n_imp)
        @test perm_id_col ≈ base_id_col atol=1e-10

        # Normalization check (UnitStd default): identified shock's impact
        # on its target variable in perm equals identified shock's impact on
        # variable 1 in base.
        @test r_perm.irf[target, target, 1] ≈ r_base.irf[1, 1, 1] atol=1e-10
    end
end

@testset "Permutation invariance: msw_confidence_set" begin
    names_base = [:Y1, :Y2, :Y3]
    model_base = fit(OLSVAR, Y, p_lag; names = names_base)
    id_base = IVIdentification(ε[:, 1], 1)
    msw_base = msw_confidence_set(model_base, id_base; horizon = 10)

    for target in 2:K
        perm = _permute_cols(K, target)
        Y_perm = Y[:, perm]
        names_perm = names_base[perm]
        model_perm = fit(OLSVAR, Y_perm, p_lag; names = names_perm)
        id_perm = IVIdentification(ε[:, 1], target)
        msw_perm = msw_confidence_set(model_perm, id_perm; horizon = 10)

        # Wald statistic is a scalar: strictly invariant
        @test msw_perm.wald_stat ≈ msw_base.wald_stat atol=1e-8
        @test msw_perm.bounded68 == msw_base.bounded68
        @test msw_perm.bounded95 == msw_base.bounded95

        # Normalization at impact: target variable's CS = {-1.0}
        @test all(msw_perm.cs68_irf_norm[1:2, 1, target] .≈ -1.0)
        @test all(msw_perm.cs95_irf_norm[1:2, 1, target] .≈ -1.0)

        # Full confidence sets (n_roots, n_imp, K). Reorder permuted-variable
        # dim back to original indexing via invperm.
        cs_perm = msw_perm.cs68_irf_norm[:, :, invperm(perm)]
        @test cs_perm ≈ msw_base.cs68_irf_norm atol=1e-6
    end
end

@testset "Permutation invariance: proxy_svar_mbb point estimates" begin
    names_base = [:Y1, :Y2, :Y3]
    seed = 1234

    model_base = fit(OLSVAR, Y, p_lag; names = names_base)
    id_base = IVIdentification(ε[:, 1], 1)
    mbb_base = proxy_svar_mbb(model_base, id_base, 8,
        ProxySVARMBB(80; block_length = 4, norm_scale = -1.0);
        rng = StableRNG(seed))

    for target in 2:K
        perm = _permute_cols(K, target)
        Y_perm = Y[:, perm]
        names_perm = names_base[perm]
        model_perm = fit(OLSVAR, Y_perm, p_lag; names = names_perm)
        id_perm = IVIdentification(ε[:, 1], target)
        mbb_perm = proxy_svar_mbb(model_perm, id_perm, 8,
            ProxySVARMBB(80; block_length = 4, norm_scale = -1.0);
            rng = StableRNG(seed))

        # point_irf shape: (K, n_imp) — single identified column flattened.
        # Reorder permuted-variable rows back to original via invperm.
        @test mbb_perm.point_irf[invperm(perm), :] ≈ mbb_base.point_irf atol=1e-10
        @test mbb_perm.point_irf_norm[invperm(perm), :] ≈ mbb_base.point_irf_norm atol=1e-10
        # Normalization: target shock's impact on target variable equals norm_scale
        @test mbb_perm.point_irf_norm[target, 1] ≈ -1.0 atol=1e-10
    end
end

@testset "Permutation invariance: irf with ProxySVARMBB bands" begin
    # axes: (variable, shock, horizon)
    names_base = [:Y1, :Y2, :Y3]
    seed = 5678

    model_base = fit(OLSVAR, Y, p_lag; names = names_base)
    id_base = IVIdentification(ε[:, 1], 1)
    r_base = irf(model_base, id_base; horizon = 6,
        inference = ProxySVARMBB(80; block_length = 4, norm_scale = -1.0),
        rng = StableRNG(seed))

    for target in 2:K
        perm = _permute_cols(K, target)
        Y_perm = Y[:, perm]
        names_perm = names_base[perm]
        model_perm = fit(OLSVAR, Y_perm, p_lag; names = names_perm)
        id_perm = IVIdentification(ε[:, 1], target)
        r_perm = irf(model_perm, id_perm; horizon = 6,
            inference = ProxySVARMBB(80; block_length = 4, norm_scale = -1.0),
            rng = StableRNG(seed))

        # Identified shock IRF column under reordering
        inv_perm = invperm(perm)
        @test Array(r_perm.irf)[inv_perm, target, :] ≈ Array(r_base.irf)[:, 1, :] atol=1e-10
        # Bands on the identified column should also match after reordering vars
        for i in eachindex(r_base.lower)
            lb_base = Array(r_base.lower[i])[:, 1, :]
            ub_base = Array(r_base.upper[i])[:, 1, :]
            lb_perm = Array(r_perm.lower[i])[inv_perm, target, :]
            ub_perm = Array(r_perm.upper[i])[inv_perm, target, :]
            @test lb_perm ≈ lb_base atol=1e-8
            @test ub_perm ≈ ub_base atol=1e-8
        end
    end
end

# ============================================================================
# Symbol target_shock resolution end-to-end
# ============================================================================
@testset "target_shock by Symbol resolves correctly" begin
    names = [:Y1, :Y2, :Y3]
    model = fit(OLSVAR, Y, p_lag; names = names)

    # ExternalInstrument by symbol ≡ by index
    id_sym = IVIdentification(ExternalInstrument(ε[:, 1]; target_shock = :Y2))
    id_idx = IVIdentification(ExternalInstrument(ε[:, 1]; target_shock = 2))

    r_sym = irf(model, id_sym; horizon = 8)
    r_idx = irf(model, id_idx; horizon = 8)
    @test Array(r_sym.irf) ≈ Array(r_idx.irf) atol=1e-12

    # ProxyIV by symbol ≡ by index
    id_proxy_sym = IVIdentification(ProxyIV(ε[:, 1]; target_shock = :Y3))
    id_proxy_idx = IVIdentification(ProxyIV(ε[:, 1]; target_shock = 3))
    r_psym = irf(model, id_proxy_sym; horizon = 8)
    r_pidx = irf(model, id_proxy_idx; horizon = 8)
    @test Array(r_psym.irf) ≈ Array(r_pidx.irf) atol=1e-12

    # Non-existent name should throw at estimation time
    id_bad = IVIdentification(ExternalInstrument(ε[:, 1]; target_shock = :nope))
    @test_throws ArgumentError irf(model, id_bad; horizon = 8)
end

# ============================================================================
# ProxyIV keyword / positional constructors
# ============================================================================
@testset "ProxyIV constructors" begin
    z = randn(100)
    # Keyword
    p1 = ProxyIV(z; target_shock = 2)
    @test p1.target_shock == 2
    # Positional (backward compat)
    p2 = ProxyIV(z, 2)
    @test p2.target_shock == 2
    # Symbol
    p3 = ProxyIV(z; target_shock = :GDP)
    @test p3.target_shock == :GDP
    # Default
    p4 = ProxyIV(z)
    @test p4.target_shock == 1
end
