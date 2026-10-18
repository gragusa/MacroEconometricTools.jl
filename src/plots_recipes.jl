# ============================================================================
# Plots Recipes for MacroEconometricTools
# ============================================================================
#
# This file defines RecipesBase recipes that will work automatically
# when Plots.jl is loaded by the user.
#

using RecipesBase: RecipesBase, @series
using Measures: mm

# ============================================================================
# Selection helpers shared by the recipe and the Makie extension
# ============================================================================

"""
    _resolve_selection(names, selection, label) -> Vector{Int}

Positions in `names` of the entries chosen by `selection`: `:all`, a vector of
names, or a vector of integer positions. Throws when an entry is not present.
"""
function _resolve_selection(names::AbstractVector, selection, label)
    selection === :all && return collect(eachindex(names))
    selection isa AbstractVector ||
        throw(
        ArgumentError(
            "`$(label)` must be `:all` or a vector of names or " *
                "positions, got $(repr(selection))"
        )
    )
    return [_resolve_entry(names, s, label) for s in selection]
end

function _resolve_entry(names::AbstractVector, i::Integer, label)
    i in eachindex(names) ||
        throw(ArgumentError("`$(label)` position $i is out of range $(eachindex(names))"))
    return Int(i)
end

function _resolve_entry(names::AbstractVector, name, label)
    idx = findfirst(isequal(name), names)
    idx === nothing &&
        throw(ArgumentError("`$(label)` entry $(repr(name)) is not one of $(names)"))
    return idx
end

"""
    _resolve_labels(labels, names, idx, suffix, label) -> Vector

Panel labels for the selected entries `names[idx]`. With `labels === nothing`
the names are used (followed by `suffix`); otherwise `labels` must have one
entry per selected name, in selection order.
"""
function _resolve_labels(labels, names::AbstractVector, idx, suffix, label)
    labels === nothing && return string.(names[idx]) .* suffix
    length(labels) == length(idx) ||
        throw(
        ArgumentError(
            "`$(label)` has $(length(labels)) labels for " *
                "$(length(idx)) selected entries"
        )
    )
    return collect(labels)
end

"""
    _prepare_irf_plot(irf; vars, shocks, pretty_vars, pretty_shocks)

Resolve the variable and shock selection and the panel labels for `irf`.
"""
function _prepare_irf_plot(
        irf::AbstractIRFResult; vars = :all, shocks = :all,
        pretty_vars = nothing, pretty_shocks = nothing
    )
    vnames = varnames(irf)
    snames = shocknames(irf)
    idxvars = _resolve_selection(vnames, vars, :vars)
    idxshocks = _resolve_selection(snames, shocks, :shocks)
    return (
        idxvars = idxvars,
        idxshocks = idxshocks,
        var_labels = _resolve_labels(pretty_vars, vnames, idxvars, "", :pretty_vars),
        shock_labels = _resolve_labels(
            pretty_shocks, snames, idxshocks, " shock", :pretty_shocks
        ),
    )
end

function _check_plot_type(irf::AbstractIRFResult, plot_type)
    plot_type ∈ (:quantiles, :paths, :both) ||
        throw(
        ArgumentError(
            "`plot_type` must be :quantiles, :paths, or :both, " *
                "got $(repr(plot_type))"
        )
    )
    plot_type === :quantiles || has_draws(irf) ||
        throw(
        ArgumentError(
            "`plot_type = $(repr(plot_type))` needs stored draws, " *
                "but this $(nameof(typeof(irf))) has none"
        )
    )
    return plot_type
end

# ============================================================================
# Recipe for any AbstractIRFResult
# ============================================================================

"""
Plot recipe for impulse responses: one panel per (variable, shock) pair with the
point estimate, confidence or credible bands (widest coverage first), and a zero
line.

Keywords: `vars`, `shocks`, `pretty_vars`, `pretty_shocks` (selection and
labels), `plot_type` (`:quantiles`, `:paths`, or `:both`; paths need stored
draws), `path_alpha`, `path_color`, `median_color` (point-estimate line, unless
`linecolor` is given), `drawzero`, `zerolinecol`.

Scale data before plotting with `rescale`.
"""
RecipesBase.@recipe function f(
        irf::AbstractIRFResult;
        vars = :all,
        shocks = :all,
        pretty_shocks = nothing,
        pretty_vars = nothing,
        plot_type = :quantiles,
        path_alpha = 0.02,
        path_color = :gray,
        median_color = :black,
        drawzero = true,
        zerolinecol = :lightgray
    )
    _check_plot_type(irf, plot_type)
    info = _prepare_irf_plot(
        irf;
        vars = vars,
        shocks = shocks,
        pretty_vars = pretty_vars,
        pretty_shocks = pretty_shocks
    )

    layout --> (length(info.idxvars), length(info.idxshocks))
    titlefontsize --> 5
    labelfontsize --> 5
    tickfontsize --> 5
    tick_direction := :none
    top_margin := -1.5mm
    label --> nothing

    H = horizon(irf)
    x = collect(0:H)
    pt_data = Array(point_estimate(irf))
    lb_data = [Array(l) for l in lowerbounds(irf)]
    ub_data = [Array(u) for u in upperbounds(irf)]
    cvgs = coverages(irf)
    band_order = sortperm(cvgs; rev = true)   # widest first, narrower on top
    draws_data = plot_type === :quantiles ? nothing : Array(get_draws(irf))

    subplot = 1
    for (row_idx, var_idx) in enumerate(info.idxvars)
        for (col_idx, shock_idx) in enumerate(info.idxshocks)
            title := info.shock_labels[col_idx]
            yguide := col_idx == 1 ? info.var_labels[row_idx] : ""
            xticks := (0:6:H, 0:6:H)
            xlims := (-0.2, H + 0.5)

            if draws_data !== nothing
                for draw_idx in axes(draws_data, 1)
                    @series begin
                        subplot := subplot
                        linecolor := path_color
                        linealpha := path_alpha
                        primary := false
                        x, draws_data[draw_idx, var_idx, shock_idx, :]
                    end
                end
            end

            if plot_type !== :paths
                for (k, ci) in enumerate(band_order)
                    @series begin
                        subplot := subplot
                        linecolor := nothing
                        fillcolor --> :red
                        fillalpha := 0.3 / k
                        primary := false
                        fillrange := lb_data[ci][var_idx, shock_idx, :]
                        x, ub_data[ci][var_idx, shock_idx, :]
                    end
                end
            end

            @series begin
                subplot := subplot
                linecolor --> median_color
                linewidth --> 2
                primary := false
                x, pt_data[var_idx, shock_idx, :]
            end

            if drawzero
                @series begin
                    subplot := subplot
                    linestyle := :dot
                    linecolor := zerolinecol
                    seriestype --> :hline
                    [0]
                end
            end

            subplot += 1
        end
    end
end
