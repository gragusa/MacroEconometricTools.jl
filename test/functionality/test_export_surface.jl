# Exported names are the end-user API; `public` names are building blocks for
# spoke packages; deprecated names are reachable only by qualification.

using Test
using MacroEconometricTools
using StableRNGs: StableRNG

const METX = MacroEconometricTools

@testset "Export surface" begin
    @testset "public, not exported" begin
        spoke_api = [
            :structural_impact, :compute_ma_matrices_from_B!, :compute_ma_matrices,
            :generate_random_orthogonal!, :check_sign_restrictions,
            :compute_structural_shocks, :check_narrative_restrictions,
            :compute_irf_point, :n_shocks, :iv_impact, :iv_instrument,
            :estimate_proxy_svar, :proxy_svar_dynamics, :bootstrap_irf,
            :refit_for_bootstrap, :proxy_svar_mbb, :msw_confidence_set,
            :lag, :create_lags!, :companion_form, :apply_constraints!,
            :build_selection_matrix, :check_constraints, :coefficient_covariance,
            :sigma_covariance, :irf_jacobian_matrices, :irf_effect_covariance,
            :irf_asymptotic_stderror, :duplication_matrix, :elimination_matrix,
            :commutation_matrix, :simulate_var, :simulate_structural,
        ]
        for name in spoke_api
            @test Base.ispublic(METX, name)
            @test !Base.isexported(METX, name)
        end
    end

    @testset "StatsAPI functions" begin
        for name in (
                :coef, :vcov, :stderror, :confint, :nobs, :dof, :dof_residual,
                :loglikelihood, :aic, :bic,
            )
            @test Base.isexported(METX, name)
            @test getfield(METX, name) === getfield(METX.StatsBase, name)
        end
    end

    @testset "deprecated names are not exported" begin
        for name in (
                :IVSVAR, :ProxyIV, :impulse_response, :effective_obs, :log_likelihood,
                :bootstrap_irf_wild, :bootstrap_irf_standard, :bootstrap_irf_block,
                :normalize, :normalize!,
            )
            @test isdefined(METX, name)
            @test !Base.isexported(METX, name)
        end
    end

    @testset "deprecated aliases forward" begin
        rng = StableRNG(1)
        Y = randn(rng, 80, 2)
        model = fit(OLSVAR, Y, 1)
        @test (@test_deprecated METX.effective_obs(model)) == nobs(model)
        expected = irf(model, CholeskyID(); horizon = 4)
        r = @test_deprecated METX.impulse_response(model, CholeskyID(); horizon = 4)
        @test point_estimate(r) == point_estimate(expected)
    end
end
