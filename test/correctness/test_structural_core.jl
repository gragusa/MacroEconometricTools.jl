# Model-level and allocating functions delegate to the matrix-level kernel in
# src/var/structural_core.jl and the in-place proxy-SVAR functions. These tests
# check the delegation against independent oracles and pin the kernel's
# argument checks.

using MacroEconometricTools
using Test
using LinearAlgebra
using Random
using StableRNGs: StableRNG

const MET = MacroEconometricTools

function _simulate_var2(rng, T, K)
    Y = zeros(T, K)
    for t in 3:T
        Y[t, :] = 0.5 * Y[t - 1, :] + 0.2 * Y[t - 2, :] + randn(rng, K)
    end
    return Y
end

@testset "Structural core delegation" begin
    rng = StableRNG(20261018)
    Y = _simulate_var2(rng, 200, 3)
    model = fit(OLSVAR, Y, 2)
    K = n_vars(model)
    Σ_L = Matrix(cholesky(vcov(model)).L)

    @testset "compute_ma_matrices matches companion powers" begin
        F = model.companion
        Φ = MET.compute_ma_matrices(F, 8, K, 2)
        @test size(Φ) == (K, K, 9)
        for h in 0:8
            @test Φ[:, :, h + 1] ≈ (F^h)[1:K, 1:K] atol = 1.0e-12
        end
        @test_throws "companion matrix must be 3×3 for n_vars=3, n_lags=1; got (6, 6)" MET.compute_ma_matrices(F, 3, K, 1)
    end

    @testset "compute_ma_matrices_from_B! argument checks" begin
        B = zeros(1 + K * 2, K)
        temp = zeros(K, K)
        @test_throws "Φ must be 3×3×H with H ≥ 5" compute_ma_matrices_from_B!(
            zeros(K, K, 4), B, temp, 4, K, 2
        )
        @test_throws "B must be 7×3" compute_ma_matrices_from_B!(
            zeros(K, K, 5), zeros(5, K), temp, 4, K, 2
        )
        @test_throws "horizon must be non-negative" compute_ma_matrices_from_B!(
            zeros(K, K, 5), B, temp, -1, K, 2
        )
        # A longer buffer is allowed; only slices 1:horizon+1 are written.
        Φ = fill(NaN, K, K, 7)
        compute_ma_matrices_from_B!(Φ, B, temp, 3, K, 2)
        @test Φ[:, :, 1] == I
        @test all(iszero, Φ[:, :, 2:4])
        @test all(isnan, Φ[:, :, 5:7])
        # Zero lags: Φ_h = 0 for h ≥ 1.
        Φ0 = fill(NaN, K, K, 4)
        compute_ma_matrices_from_B!(Φ0, zeros(1, K), temp, 3, K, 0)
        @test Φ0[:, :, 1] == I
        @test all(iszero, Φ0[:, :, 2:4])
    end

    @testset "rotation_matrix(::CholeskyID) via structural_impact" begin
        @test rotation_matrix(model, CholeskyID()) == structural_impact(Σ_L, CholeskyID())
        for ordering in ([3, 1, 2], [2, 3, 1])
            P = rotation_matrix(model, CholeskyID(ordering))
            # ordering_idx defaults to the integer ordering stored in the id
            @test structural_impact(Σ_L, CholeskyID(ordering)) == P
            # Oracle: Cholesky of the reordered covariance, permuted back
            Σ = Matrix(vcov(model))
            L = cholesky(Symmetric(Σ[ordering, ordering])).L
            inv_perm = invperm(ordering)
            @test P ≈ L[inv_perm, inv_perm] rtol = 1.0e-10
        end
        names_order = model.names[[3, 1, 2]]
        @test rotation_matrix(model, CholeskyID(names_order)) ≈
            rotation_matrix(model, CholeskyID([3, 1, 2])) rtol = 1.0e-12
        @test structural_impact(
            Σ_L, CholeskyID(names_order); ordering_idx = [3, 1, 2]
        ) == rotation_matrix(model, CholeskyID(names_order))
        @test_throws "given by name; pass ordering_idx" structural_impact(
            Σ_L, CholeskyID(names_order)
        )
        @test_throws "contradicts CholeskyID ordering" structural_impact(
            Σ_L, CholeskyID([3, 1, 2]); ordering_idx = [1, 2, 3]
        )
        @test_throws "contradicts CholeskyID ordering" structural_impact(
            Σ_L, CholeskyID([3, 1, 2]); ordering_idx = nothing
        )
    end

    @testset "generate_random_orthogonal wraps the in-place form" begin
        for n in 1:5
            Q = MET.generate_random_orthogonal(n, StableRNG(n))
            Q! = generate_random_orthogonal!(zeros(n, n), zeros(n, n), StableRNG(n))
            @test Q == Q!
            @test Q' * Q ≈ I atol = 1.0e-12
        end
    end

    @testset "check_sign_restrictions: model level equals matrix level" begin
        restrictions = [1 0 0; 0 -1 0; 0 0 0]
        Φ = MET.compute_ma_matrices(model.companion, 4, K, 2)
        rng_q = StableRNG(7)
        agree = map(1:200) do _
            P = Σ_L * MET.generate_random_orthogonal(K, rng_q)
            check_sign_restrictions(P, restrictions, model, 4) ==
                check_sign_restrictions(P, restrictions, Φ, 4)
        end
        @test all(agree)
        @test_throws "Φ holds horizons 0:4; restrictions need 0:5" check_sign_restrictions(
            Σ_L, restrictions, Φ, 5
        )
        @test_throws "restrictions must be 3×3" check_sign_restrictions(
            Σ_L, [1 0; 0 1], Φ, 0
        )
    end

    @testset "estimate_proxy_svar wraps the in-place QR estimator" begin
        # Persistent data at a large level: cond(X) ≈ 1e6, where normal
        # equations lose about four digits relative to QR.
        rng_p = StableRNG(1)
        Z = cumsum(randn(rng_p, 300, 3), dims = 1) .+ 1000
        p = 4
        yy = Z[(p + 1):end, :]
        xx = hcat(ones(size(yy, 1)), (Z[(p + 1 - l):(end - l), :] for l in 1:p)...)
        mm = randn(rng_p, size(yy, 1))

        A_est, U_est, Σ_uu, Σ_um, H1 = MET.estimate_proxy_svar(yy, xx, mm)
        A_big = (big.(xx)' * big.(xx)) \ (big.(xx)' * big.(yy))
        @test maximum(abs, A_est' .- A_big) < 1.0e-10

        buf = MET.ProxySvarBuffers{Float64}(size(yy, 1), size(yy, 2), size(xx, 2))
        out! = MET.estimate_proxy_svar!(buf, yy, xx, mm)
        @test all(map(==, (A_est, U_est, Σ_uu, Σ_um, H1), out!))
        @test U_est ≈ yy - xx * A_est' atol = 1.0e-9
        @test H1 ≈ Σ_um / sqrt(dot(Σ_um, Σ_uu \ Σ_um))

        collinear = copy(xx)
        collinear[:, 3] .= collinear[:, 2]
        @test_throws SingularException MET.estimate_proxy_svar(yy, collinear, mm)
        @test_throws "buffers sized for xx" MET.estimate_proxy_svar!(
            buf, yy[1:(end - 1), :], xx[1:(end - 1), :], mm[1:(end - 1)]
        )
        @test_throws "at least as many observations" MET.ProxySvarBuffers{Float64}(3, 2, 5)
    end

    @testset "proxy_svar_dynamics wraps the in-place form" begin
        rng_d = StableRNG(3)
        Kd, p, n_imp = 3, 2, 10
        A_est = 0.1 * randn(rng_d, Kd, 1 + Kd * p)
        Σ_uu = Matrix(Symmetric(I + 0.1 * ones(Kd, Kd)))
        Σ_um = [0.5, 0.2, -0.1]
        H1 = Σ_um / sqrt(dot(Σ_um, Σ_uu \ Σ_um))
        dyn = MET.proxy_svar_dynamics(A_est, Σ_uu, Σ_um, H1, p, -1.0, n_imp, 1)
        buf = MET.ProxyDynamicsBuffers{Float64}(Kd, p, n_imp)
        MET.proxy_svar_dynamics!(buf, A_est, Σ_uu, Σ_um, H1, p, -1.0, n_imp, 1)
        @test dyn.irf == buf.irf && dyn.irf_norm == buf.irf_norm &&
            dyn.fevd == buf.fevd && dyn.svma == buf.svma

        # Oracle: IRF_h = Φ_h H1 from the hub's MA recursion
        B = vcat(A_est[:, 1]', A_est[:, 2:end]')
        Φ = compute_ma_matrices_from_B!(
            zeros(Kd, Kd, n_imp), B, zeros(Kd, Kd), n_imp - 1, Kd, p
        )
        for h in 1:n_imp
            @test dyn.irf[:, h] ≈ Φ[:, :, h] * H1 atol = 1.0e-12
        end

        @test_throws "buffers sized for" MET.proxy_svar_dynamics!(
            buf, A_est, Σ_uu, Σ_um, H1, p, -1.0, n_imp + 1, 1
        )
        @test_throws "target must lie in 1:3" MET.proxy_svar_dynamics(
            A_est, Σ_uu, Σ_um, H1, p, -1.0, n_imp, 4
        )
    end
end
