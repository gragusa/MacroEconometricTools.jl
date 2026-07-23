using Test
using MacroEconometricTools
import RecipesBase
using AxisArrays: AxisArray, Axis
using StableRNGs: StableRNG

# RecipesBase delegates supported-key knowledge to plotting backends. Defining
# this hook is its documented backend-free recipe testing pattern.
RecipesBase.is_key_supported(::Symbol) = true

function recipe_fixture_arrays()
    data = AxisArray(reshape(collect(1.0:10.0), 2, 1, 5),
        Axis{:variable}([:y, :x]), Axis{:shock}([:s]), Axis{:horizon}(0:4))
    lower = [AxisArray(Array(data) .- 0.2,
        Axis{:variable}([:y, :x]), Axis{:shock}([:s]), Axis{:horizon}(0:4))]
    upper = [AxisArray(Array(data) .+ 0.2,
        Axis{:variable}([:y, :x]), Axis{:shock}([:s]), Axis{:horizon}(0:4))]
    return data, lower, upper
end

@testset "IRF plot recipes without a plotting backend" begin
    rng = StableRNG(91)
    model = fit(OLSVAR, randn(rng, 100, 2), 1; names = [:y, :x])
    frequentist = irf(model, CholeskyID(); horizon = 4, inference = Analytic())
    series = RecipesBase.apply_recipe(Dict{Symbol, Any}(:vars => [:y]), frequentist)
    @test !isempty(series)
    @test first(series).plotattributes[:layout] == (1, 2)
    @test_throws ErrorException RecipesBase.apply_recipe(
        Dict{Symbol, Any}(:vars => [:missing]), frequentist)

    data, lower, upper = recipe_fixture_arrays()
    draws = AxisArray(repeat(reshape(Array(data), 1, 2, 1, 5), 3, 1, 1, 1),
        Axis{:draw}(1:3), Axis{:variable}([:y, :x]), Axis{:shock}([:s]),
        Axis{:horizon}(0:4))
    bayesian = BayesianIRFResult{Float64, typeof(draws), CholeskyID}(
        draws, lower, upper, [0.9], CholeskyID(), (;))
    @test !isempty(RecipesBase.apply_recipe(
        Dict{Symbol, Any}(:plot_type => :both), bayesian))

    lp_data = AxisArray(reshape(collect(1.0:10.0), 2, 1, 5),
        Axis{:response}([:y, :x]), Axis{:shock}([:s]), Axis{:horizon}(0:4))
    lp_stderr = AxisArray(fill(0.1, 2, 1, 5),
        Axis{:response}([:y, :x]), Axis{:shock}([:s]), Axis{:horizon}(0:4))
    lp_lower = [AxisArray(Array(lp_data) .- 0.2,
        Axis{:response}([:y, :x]), Axis{:shock}([:s]), Axis{:horizon}(0:4))]
    lp_upper = [AxisArray(Array(lp_data) .+ 0.2,
        Axis{:response}([:y, :x]), Axis{:shock}([:s]), Axis{:horizon}(0:4))]
    local_projection = LocalProjectionIRFResult{Float64, typeof(lp_data)}(
        lp_data, lp_stderr, lp_lower, lp_upper, [0.9], (;))
    @test !isempty(RecipesBase.apply_recipe(Dict{Symbol, Any}(), local_projection))
end
