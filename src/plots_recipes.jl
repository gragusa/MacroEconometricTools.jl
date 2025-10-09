# ============================================================================
# Plots Recipes for MacroEconometricTools
# ============================================================================
#
# This file defines RecipesBase recipes that will work automatically
# when Plots.jl is loaded by the user.
#

using RecipesBase

const _IRFResult = IRFResult

lowerbounds(irf::_IRFResult) = irf.lower
upperbounds(irf::_IRFResult) = irf.upper

function _var_names(irf::_IRFResult)
    if haskey(irf.metadata, :names)
        names = irf.metadata.names
        return Symbol.(names)
    else
        return [Symbol("Y_$i") for i in 1:MacroEconometricTools.n_vars(irf)]
    end
end

function _resolve_indices(nms::Vector{Symbol}, selection, label)
    if selection === :all
        return collect(1:length(nms))
    elseif selection isa AbstractVector{Symbol}
        idx = findall(x -> x ∈ selection, nms)
        length(idx) == length(selection) ||
            error("At least one $(label) entry is not present in the IRF names")
        return idx
    else
        error("`$(label)` must be either a vector of symbols or :all, got $(selection)")
    end
end

function _resolve_labels(labels, nms::Vector{Symbol}, suffix::AbstractString)
    if labels === nothing
        base = String.(nms)
        return suffix === "" ? base : base .* suffix
    elseif length(labels) != length(nms)
        error("Label vector must have the same length as the number of variables in the IRF")
    else
        return labels
    end
end

function _prepare_irf_plot(irf::_IRFResult; vars=:all, shocks=:all,
    pretty_vars=nothing, pretty_shocks=nothing)
    nms = _var_names(irf)
    idxvars = _resolve_indices(nms, vars, :vars)
    idxshocks = _resolve_indices(nms, shocks, :shocks)
    var_labels_full = _resolve_labels(pretty_vars, nms, "")
    shock_labels_full = _resolve_labels(pretty_shocks, nms, " shock")
    return (
        idxvars = idxvars,
        idxshocks = idxshocks,
        var_labels = var_labels_full[idxvars],
        shock_labels = shock_labels_full[idxshocks]
    )
end

RecipesBase.@recipe function f(irf::_IRFResult;
    vars=:all,
    shocks=:all,
    pretty_shocks=nothing,
    pretty_vars=nothing,
    drawzero=true,
    zerolinecol=:lightgray)

    info = _prepare_irf_plot(irf;
        vars=vars,
        shocks=shocks,
        pretty_vars=pretty_vars,
        pretty_shocks=pretty_shocks
    )

    layout --> (length(info.idxvars), length(info.idxshocks))
    linecolor --> :black
    titlefontsize --> 5
    labelfontsize --> 5
    tickfontsize --> 5
    tick_direction := :none
    top_margin := -1.5mm
    label --> nothing

    subplot = 1

    lb = lowerbounds(irf)
    ub = upperbounds(irf)
    coverages = irf.coverage

    for (row_idx, var_idx) in enumerate(info.idxvars)
        for (col_idx, shock_idx) in enumerate(info.idxshocks)
            y = irf.irf[:, var_idx, shock_idx]
            x = 0:(length(y)-1)
            title := info.shock_labels[col_idx]
            yguide := col_idx == 1 ? info.var_labels[row_idx] : ""
            xticks := (0:6:length(y), 0:6:length(y))
            xlims := (-0.2, length(y))
            for (cv_idx, _) in enumerate(coverages)
                lb_slice = lb[cv_idx][:, var_idx, shock_idx]
                ub_slice = ub[cv_idx][:, var_idx, shock_idx]
                @series begin
                    subplot := subplot
                    linecolor := nothing
                    fillcolor --> :red
                    fillalpha := 0.5 / (1.2 * cv_idx)
                    primary := false
                    fillrange := lb_slice
                    x, ub_slice
                end
            end
            @series begin
                subplot := subplot
                x, y
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

# ============================================================================
# Recipe for SignRestrictedIRFResult
# ============================================================================

const _SignRestrictedIRFResult = SignRestrictedIRFResult

function _var_names_sign(irf::_SignRestrictedIRFResult)
    if haskey(irf.metadata, :names)
        return Symbol.(irf.metadata.names)
    else
        n_vars = size(irf.irf_median, 2)
        return [Symbol("Y_$i") for i in 1:n_vars]
    end
end

"""
Plot recipe for sign-restricted IRFs.

Supports three plot types:
- `:paths` - Show all drawn IRF paths with transparency
- `:quantiles` - Show pointwise quantile bands
- `:both` - Show both paths and quantiles
"""
RecipesBase.@recipe function f(irf::_SignRestrictedIRFResult;
    vars=:all,
    shocks=:all,
    pretty_shocks=nothing,
    pretty_vars=nothing,
    plot_type=:quantiles,
    path_alpha=0.02,
    path_color=:gray,
    median_color=:black,
    drawzero=true,
    zerolinecol=:lightgray)

    nms = _var_names_sign(irf)
    idxvars = _resolve_indices(nms, vars, :vars)
    idxshocks = _resolve_indices(nms, shocks, :shocks)
    var_labels_full = _resolve_labels(pretty_vars, nms, "")
    shock_labels_full = _resolve_labels(pretty_shocks, nms, " shock")

    layout --> (length(idxvars), length(idxshocks))
    titlefontsize --> 5
    labelfontsize --> 5
    tickfontsize --> 5
    tick_direction := :none
    top_margin := -1.5mm
    label --> nothing

    subplot = 1
    lb = irf.lower
    ub = irf.upper
    coverages = irf.coverage

    for (row_idx, var_idx) in enumerate(idxvars)
        for (col_idx, shock_idx) in enumerate(idxshocks)
            y_median = irf.irf_median[:, var_idx, shock_idx]
            x = 0:(length(y_median)-1)
            title := shock_labels_full[shock_idx]
            yguide := col_idx == 1 ? var_labels_full[row_idx] : ""
            xticks := (0:6:length(y_median), 0:6:length(y_median))
            xlims := (-0.2, length(y_median))

            # Draw paths if requested
            if plot_type ∈ [:paths, :both]
                n_draws = size(irf.irf_draws, 1)
                for draw_idx in 1:n_draws
                    y_path = irf.irf_draws[draw_idx, :, var_idx, shock_idx]
                    @series begin
                        subplot := subplot
                        linecolor := path_color
                        linealpha := path_alpha
                        primary := false
                        x, y_path
                    end
                end
            end

            # Draw quantile bands if requested
            if plot_type ∈ [:quantiles, :both]
                for (cv_idx, _) in enumerate(coverages)
                    lb_slice = lb[cv_idx][:, var_idx, shock_idx]
                    ub_slice = ub[cv_idx][:, var_idx, shock_idx]
                    @series begin
                        subplot := subplot
                        linecolor := nothing
                        fillcolor --> :red
                        fillalpha := 0.5 / (1.2 * cv_idx)
                        primary := false
                        fillrange := lb_slice
                        x, ub_slice
                    end
                end
            end

            # Draw median
            @series begin
                subplot := subplot
                linecolor := median_color
                linewidth := 2
                primary := false
                x, y_median
            end

            # Zero line
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
