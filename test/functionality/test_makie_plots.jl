using Test
using MacroEconometricTools
using StableRNGs: StableRNG

makie_ext() = Base.get_extension(MacroEconometricTools, :MacroEconometricToolsMakieExt)

function makie_fixture()
    rng = StableRNG(17)
    model = fit(OLSVAR, randn(rng, 120, 3), 2; names = [:y, :π, :i])
    analytic = irf(model, CholeskyID(); horizon = 8, inference = Analytic())
    bootstrap = irf(
        model, CholeskyID(); horizon = 8,
        inference = WildBootstrap(20; save_draws = true), rng
    )
    return analytic, bootstrap
end

# The hint is only printed while the Makie extension is not loaded, so this
# block runs only in a session that has not loaded a Makie backend yet.
if makie_ext() === nothing
    @testset "irfplot without a Makie backend" begin
        analytic, _ = makie_fixture()
        for (f, args) in ((irfplot, (analytic,)), (irfplot!, (nothing, analytic)))
            err = try
                f(args...)
            catch e
                e
            end
            @test err isa MethodError
            msg = sprint(showerror, err)
            @test occursin("implemented by the Makie extension", msg)
            @test occursin("using CairoMakie", msg)
        end
    end
end

using CairoMakie: CairoMakie

@testset "Makie irfplot / irfplot!" begin
    @test makie_ext() !== nothing
    # With the extension loaded, a MethodError carries no backend hint.
    msg = sprint(
        showerror, try
            irfplot(1)
        catch e
            e
        end
    )
    @test !occursin("Makie extension", msg)

    analytic, bootstrap = makie_fixture()
    axes_of(fig) = [c for c in fig.content if c isa CairoMakie.Axis]

    fig = irfplot(analytic)
    @test fig isa CairoMakie.Figure
    @test length(axes_of(fig)) == 9

    fig = irfplot(
        bootstrap; vars = [:π, :i], shocks = [1], plot_type = :both,
        pretty_vars = ["Inflation", "Rate"], zerolinecolor = :red, title = "IRFs"
    )
    @test length(axes_of(fig)) == 2
    @test [ax.ylabel[] for ax in axes_of(fig)] == ["Inflation", "Rate"]
    path = joinpath(mktempdir(), "irf.png")
    CairoMakie.save(path, fig)
    @test filesize(path) > 0

    fig = CairoMakie.Figure()
    ax = CairoMakie.Axis(fig[1, 1])
    @test irfplot!(ax, bootstrap; var = :y, shock = 2, plot_type = :paths) === ax
    # 20 draw paths, the point estimate, and the zero line
    @test length(ax.scene.plots) == 22

    @test_throws "needs stored draws" irfplot(analytic; plot_type = :paths)
    @test_throws "entry :x is not one of" irfplot!(
        CairoMakie.Axis(CairoMakie.Figure()[1, 1]), analytic; var = :x, shock = 1)
end
