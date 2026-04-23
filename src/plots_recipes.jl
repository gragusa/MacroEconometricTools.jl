# ============================================================================
# Plots Recipes for MacroEconometricTools
# ============================================================================
#
# This file defines RecipesBase recipes that will work automatically
# when Plots.jl is loaded by the user.
#

using RecipesBase
using AxisArrays: AxisArrays, Axis

# Note: lowerbounds, upperbounds, coverages are now defined in types.jl
# They work on any AbstractIRFResult

function _var_names(irf::AbstractIRFResult)
    try
        return varnames(irf)
    catch
        if haskey(irf.metadata, :names)
            return Symbol.(irf.metadata.names)
        else
            return [Symbol("Y_$i") for i in 1:MacroEconometricTools.n_vars(irf)]
        end
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

function _prepare_irf_plot(irf::AbstractIRFResult; vars = :all, shocks = :all,
        pretty_vars = nothing, pretty_shocks = nothing)
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

RecipesBase.@recipe function f(irf::IRFResult;
        vars = :all,
        shocks = :all,
        pretty_shocks = nothing,
        pretty_vars = nothing,
        drawzero = true,
        zerolinecol = :lightgray)
    info = _prepare_irf_plot(irf;
        vars = vars,
        shocks = shocks,
        pretty_vars = pretty_vars,
        pretty_shocks = pretty_shocks
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

    # Materialize AxisArrays to plain arrays for indexing
    irf_data = Array(irf.irf)       # (variable, shock, horizon)
    lb_data = [Array(l) for l in lb]
    ub_data = [Array(u) for u in ub]

    for (row_idx, var_idx) in enumerate(info.idxvars)
        for (col_idx, shock_idx) in enumerate(info.idxshocks)
            y = irf_data[var_idx, shock_idx, :]
            x = 0:(length(y) - 1)
            title := info.shock_labels[col_idx]
            yguide := col_idx == 1 ? info.var_labels[row_idx] : ""
            xticks := (0:6:length(y), 0:6:length(y))
            xlims := (-0.2, length(y))
            for (cv_idx, _) in enumerate(coverages)
                lb_slice = lb_data[cv_idx][var_idx, shock_idx, :]
                ub_slice = ub_data[cv_idx][var_idx, shock_idx, :]
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

"""
Plot recipe for sign-restricted IRFs.

Supports three plot types:
- `:paths` - Show all drawn IRF paths with transparency
- `:quantiles` - Show pointwise quantile bands
- `:both` - Show both paths and quantiles
"""
RecipesBase.@recipe function f(irf::SignRestrictedIRFResult;
        vars = :all,
        shocks = :all,
        pretty_shocks = nothing,
        pretty_vars = nothing,
        plot_type = :quantiles,
        path_alpha = 0.02,
        path_color = :gray,
        median_color = :black,
        drawzero = true,
        zerolinecol = :lightgray)

    # Use unified _prepare_irf_plot helper
    info = _prepare_irf_plot(irf;
        vars = vars,
        shocks = shocks,
        pretty_vars = pretty_vars,
        pretty_shocks = pretty_shocks
    )

    idxvars = info.idxvars
    idxshocks = info.idxshocks
    var_labels_full = info.var_labels
    shock_labels_full = info.shock_labels

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

    # Materialize AxisArrays to plain arrays for indexing
    median_data = Array(irf.irf_median)   # (variable, shock, horizon)
    draws_data = Array(irf.irf_draws)     # (draw, variable, shock, horizon)
    lb_data = [Array(l) for l in lb]
    ub_data = [Array(u) for u in ub]

    for (row_idx, var_idx) in enumerate(idxvars)
        for (col_idx, shock_idx) in enumerate(idxshocks)
            y_median = median_data[var_idx, shock_idx, :]
            x = 0:(length(y_median) - 1)
            title := shock_labels_full[col_idx]
            yguide := col_idx == 1 ? var_labels_full[row_idx] : ""
            xticks := (0:6:length(y_median), 0:6:length(y_median))
            xlims := (-0.2, length(y_median))

            # Draw paths if requested
            if plot_type ∈ [:paths, :both]
                nd = size(draws_data, 1)
                for draw_idx in 1:nd
                    y_path = draws_data[draw_idx, var_idx, shock_idx, :]
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
                    lb_slice = lb_data[cv_idx][var_idx, shock_idx, :]
                    ub_slice = ub_data[cv_idx][var_idx, shock_idx, :]
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

# ============================================================================
# Recipe for BayesianIRFResult (AxisArray-based)
# ============================================================================

"""
Plot recipe for Bayesian IRFs with posterior draws.

Supports:
- `plot_type`: `:quantiles`, `:paths`, or `:both`
- Multiple coverage levels with layered ribbons

Scale data before plotting with `rescale` / `rescale!`.
"""
RecipesBase.@recipe function f(irf::BayesianIRFResult;
        vars = :all,
        shocks = :all,
        pretty_shocks = nothing,
        pretty_vars = nothing,
        plot_type = :quantiles,
        path_alpha = 0.02,
        path_color = :gray,
        median_color = :black,
        drawzero = true,
        zerolinecol = :lightgray)

    # Get variable and shock names from AxisArray axes
    var_axis = AxisArrays.axes(irf.data, Axis{:variable})
    shock_axis = AxisArrays.axes(irf.data, Axis{:shock})
    horizon_axis = AxisArrays.axes(irf.data, Axis{:horizon})

    all_vars = collect(AxisArrays.axisvalues(var_axis)[1])
    all_shocks = collect(AxisArrays.axisvalues(shock_axis)[1])
    horizons = collect(AxisArrays.axisvalues(horizon_axis)[1])

    # Resolve variable and shock indices
    idxvars = vars === :all ? (1:length(all_vars)) :
              _resolve_indices_generic(all_vars, vars)
    idxshocks = shocks === :all ? (1:length(all_shocks)) :
                _resolve_indices_generic(all_shocks, shocks)

    var_labels = pretty_vars === nothing ? string.(all_vars[idxvars]) : pretty_vars
    shock_labels = pretty_shocks === nothing ? string.(all_shocks[idxshocks]) .* " shock" :
                   pretty_shocks

    layout --> (length(idxvars), length(idxshocks))
    titlefontsize --> 5
    labelfontsize --> 5
    tickfontsize --> 5
    tick_direction := :none
    top_margin := -1.5mm
    label --> nothing

    subplot = 1
    lb = lowerbounds(irf)
    ub = upperbounds(irf)
    cvgs = coverages(irf)

    # Get point estimate (median across draws)
    pt_est = point_estimate(irf)

    for (row_idx, var_idx) in enumerate(idxvars)
        for (col_idx, shock_idx) in enumerate(idxshocks)
            # Extract point estimate for this variable/shock
            y_median = Array(pt_est)[var_idx, shock_idx, :]
            x = horizons
            title := shock_labels[col_idx]
            yguide := col_idx == 1 ? var_labels[row_idx] : ""
            xticks := (0:6:maximum(horizons), 0:6:maximum(horizons))
            xlims := (-0.2, maximum(horizons) + 0.5)

            # Draw paths if requested
            if plot_type ∈ [:paths, :both]
                n_drw = n_draws(irf)
                data_arr = Array(irf.data)
                for draw_idx in 1:n_drw
                    y_path = data_arr[draw_idx, var_idx, shock_idx, :]
                    @series begin
                        subplot := subplot
                        linecolor := path_color
                        linealpha := path_alpha
                        primary := false
                        x, y_path
                    end
                end
            end

            # Draw quantile bands if requested (widest first for proper stacking)
            if plot_type ∈ [:quantiles, :both]
                for (cv_idx, _) in enumerate(reverse(cvgs))
                    rev_idx = length(cvgs) - cv_idx + 1
                    lb_slice = Array(lb[rev_idx])[var_idx, shock_idx, :]
                    ub_slice = Array(ub[rev_idx])[var_idx, shock_idx, :]
                    @series begin
                        subplot := subplot
                        linecolor := nothing
                        fillcolor --> :red
                        fillalpha := 0.3 / cv_idx  # Darker for narrower bands
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

# ============================================================================
# Recipe for LocalProjectionIRFResult (AxisArray-based)
# ============================================================================

"""
Plot recipe for Local Projection IRFs.

Scale data before plotting with `rescale` / `rescale!`.
"""
RecipesBase.@recipe function f(irf::LocalProjectionIRFResult;
        vars = :all,
        shocks = :all,
        pretty_vars = nothing,
        pretty_shocks = nothing,
        drawzero = true,
        zerolinecol = :lightgray,
        linecolor = :black)

    # Get dimensions from AxisArray axes
    response_axis = AxisArrays.axes(irf.data, Axis{:response})
    shock_axis = AxisArrays.axes(irf.data, Axis{:shock})
    horizon_axis = AxisArrays.axes(irf.data, Axis{:horizon})

    all_responses = collect(AxisArrays.axisvalues(response_axis)[1])
    all_shocks = collect(AxisArrays.axisvalues(shock_axis)[1])
    horizons = collect(AxisArrays.axisvalues(horizon_axis)[1])

    idxvars = vars === :all ? collect(1:length(all_responses)) :
              _resolve_indices_generic(all_responses, vars)
    idxshocks = shocks === :all ? collect(1:length(all_shocks)) :
                _resolve_indices_generic(all_shocks, shocks)

    var_labels = pretty_vars === nothing ? string.(all_responses[idxvars]) : pretty_vars
    shock_labels = pretty_shocks === nothing ?
                   string.(all_shocks[idxshocks]) .* " shock" : pretty_shocks

    layout --> (length(idxvars), length(idxshocks))
    titlefontsize --> 5
    labelfontsize --> 5
    tickfontsize --> 5
    tick_direction := :none
    top_margin := -1.5mm
    label --> nothing

    subplot = 1
    lb = lowerbounds(irf)
    ub = upperbounds(irf)
    cvgs = coverages(irf)

    pt_data = Array(irf.data)

    for (ri, vi) in enumerate(idxvars)
        for (ci, si) in enumerate(idxshocks)
            y = pt_data[vi, si, :]
            x = horizons
            title := shock_labels[ci]
            yguide := ci == 1 ? var_labels[ri] : ""
            xticks := (0:6:maximum(horizons), 0:6:maximum(horizons))
            xlims := (-0.2, maximum(horizons) + 0.5)

            # Draw confidence bands (widest first)
            for (cv_idx, _) in enumerate(reverse(cvgs))
                rev_idx = length(cvgs) - cv_idx + 1
                lb_slice = Array(lb[rev_idx])[vi, si, :]
                ub_slice = Array(ub[rev_idx])[vi, si, :]
                @series begin
                    subplot := subplot
                    linecolor := nothing
                    fillcolor --> :blue
                    fillalpha := 0.3 / cv_idx
                    primary := false
                    fillrange := lb_slice
                    x, ub_slice
                end
            end

            # Point estimate line
            @series begin
                subplot := subplot
                linecolor := linecolor
                linewidth := 2
                x, y
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

# Helper for generic index resolution with AxisArray values
function _resolve_indices_generic(all_names, selection)
    if selection isa AbstractVector{Symbol}
        return [findfirst(==(s), all_names) for s in selection]
    elseif selection isa AbstractVector{<:Integer}
        return selection
    else
        error("Selection must be :all, a vector of Symbols, or a vector of integers")
    end
end
