module MacroEconometricToolsMakieExt

using Makie
using MacroEconometricTools
using AxisArrays: AxisArrays, Axis

import MacroEconometricTools: irfplot, irfplot!

const _IRFResult = MacroEconometricTools.IRFResult
const _SignRestrictedIRFResult = MacroEconometricTools.SignRestrictedIRFResult
const _BayesianIRFResult = MacroEconometricTools.BayesianIRFResult
const _LocalProjectionIRFResult = MacroEconometricTools.LocalProjectionIRFResult
const _AbstractIRFResult = MacroEconometricTools.AbstractIRFResult

const lowerbounds = MacroEconometricTools.lowerbounds
const upperbounds = MacroEconometricTools.upperbounds
const coverages = MacroEconometricTools.coverages
const point_estimate = MacroEconometricTools.point_estimate
const n_draws = MacroEconometricTools.n_draws

# ============================================================================
# Name / index resolution helpers
# ============================================================================

function _var_names(irf::Union{_IRFResult, _SignRestrictedIRFResult})
    if haskey(irf.metadata, :names)
        return Symbol.(irf.metadata.names)
    else
        return [Symbol("Y_$i") for i in 1:MacroEconometricTools.n_vars(irf)]
    end
end

function _resolve_indices(nms::Vector{Symbol}, selection, label)
    selection === :all && return collect(1:length(nms))
    if selection isa AbstractVector{Symbol}
        idx = findall(x -> x ∈ selection, nms)
        length(idx) == length(selection) ||
            error("At least one $(label) entry is not present in the IRF names")
        return idx
    end
    error("`$(label)` must be either a vector of symbols or :all, got $(selection)")
end

function _resolve_labels(labels, nms::Vector{Symbol}, suffix::AbstractString)
    labels === nothing && return (suffix == "" ? String.(nms) : String.(nms) .* suffix)
    length(labels) == length(nms) ||
        error("Label vector must have the same length as the number of variables in the IRF")
    return labels
end

function _prepare_irf_plot(
        irf::Union{_IRFResult, _SignRestrictedIRFResult}; vars = :all, shocks = :all,
        pretty_vars = nothing, pretty_shocks = nothing)
    nms = _var_names(irf)
    idxvars = _resolve_indices(nms, vars, :vars)
    idxshocks = _resolve_indices(nms, shocks, :shocks)
    var_labels_full = _resolve_labels(pretty_vars, nms, "")
    shock_labels_full = _resolve_labels(pretty_shocks, nms, " shock")
    return (
        idxvars = idxvars, idxshocks = idxshocks,
        var_labels = var_labels_full[idxvars],
        shock_labels = shock_labels_full[idxshocks])
end

function _resolve_indices_generic(all_names, selection)
    if selection isa AbstractVector{Symbol}
        return [findfirst(==(s), all_names) for s in selection]
    elseif selection isa AbstractVector{<:Integer}
        return selection
    else
        error("Selection must be :all, a vector of Symbols, or a vector of integers")
    end
end

# ============================================================================
# Shared panel-drawing core
# ============================================================================

"""
    _plot_irf_panel!(ax, xvals, y, lb, ub, cvgs; <keyword arguments>)

Draw a single IRF panel: confidence/credible bands (widest first), point
estimate line, and optional zero reference line.

`lb` and `ub` are vectors of vectors (one per coverage level), already sliced
to the correct variable/shock combination.  All scaling must be applied to the
data *before* calling this function (use `rescale` / `rescale!`).
"""
function _plot_irf_panel!(ax, xvals, y, lb, ub, cvgs;
        drawzero = true, zerolinecolor = :gray60, zerolinestyle = :dash,
        bandcolor = :steelblue, bandalpha = 0.2,
        linecolor = :black, linewidth = 2.0,
        xtickstep = 4)
    Makie.xlims!(ax, xvals[1] - 0.4, xvals[end] + 0.4)
    if xtickstep > 0
        ax.xticks = xvals[1]:xtickstep:xvals[end]
    end

    # Bands — widest (highest coverage) first so narrower bands paint on top
    if !isempty(cvgs) && !isempty(lb)
        order = sortperm(cvgs; rev = true)          # widest first
        for (draw_order, ci) in enumerate(order)
            lower = lb[ci]
            upper = ub[ci]
            # Darker for narrower bands
            alpha = clamp(bandalpha + 0.12f0 * (draw_order - 1), 0.0f0, 0.9f0)
            color = Makie.RGBAf(Makie.to_color(bandcolor), alpha)
            Makie.band!(ax, xvals, lower, upper; color = color)
        end
    end

    # Point estimate
    Makie.lines!(ax, xvals, y;
        color = linecolor, linewidth = linewidth)

    # Zero reference
    if drawzero
        Makie.hlines!(ax, [0.0]; color = zerolinecolor,
            linewidth = 1, linestyle = zerolinestyle)
    end
end

"""
    _plot_paths!(ax, xvals, draws; path_alpha, path_color, path_linewidth)

Overlay individual IRF draws as faint lines (sign-restricted / Bayesian).
"""
function _plot_paths!(ax, xvals, draws::AbstractMatrix;
        path_alpha = 0.02, path_color = :gray,
        path_linewidth = 0.5)
    col = Makie.RGBAf(Makie.to_color(path_color), Float32(path_alpha))
    for i in axes(draws, 1)
        Makie.lines!(ax, xvals, view(draws, i, :);
            color = col, linewidth = path_linewidth)
    end
end

# ============================================================================
# Figure / layout helpers
# ============================================================================

function _make_figure(; figure = (;), size = nothing, title = nothing,
        title_fontsize = 20, title_font = :bold)
    fig_kw = Dict{Symbol, Any}(pairs(figure)...)
    if size !== nothing
        fig_kw[:size] = size
    end
    fig = Makie.Figure(; fig_kw...)
    if title !== nothing
        Makie.Label(fig[0, :]; text = title, fontsize = title_fontsize,
            font = title_font)
    end
    return fig
end

function _configure_axes!(fig, axes_matrix, nrows, ncols;
        linkxaxes = true, linkyaxes = :row,
        colgap = 10, rowgap = 10)
    # Link x-axes across all panels
    if linkxaxes
        all_axes = vec(axes_matrix)
        length(all_axes) > 1 && Makie.linkxaxes!(all_axes...)
    end
    # Link y-axes per row
    if linkyaxes === :row
        for r in 1:nrows
            row_axes = axes_matrix[r, :]
            length(row_axes) > 1 && Makie.linkyaxes!(row_axes...)
        end
    elseif linkyaxes === true || linkyaxes === :all
        all_axes = vec(axes_matrix)
        length(all_axes) > 1 && Makie.linkyaxes!(all_axes...)
    end
    # Hide decorations on interior panels
    for r in 1:nrows, c in 1:ncols

        ax = axes_matrix[r, c]
        c > 1 && (ax.yticklabelsvisible = false; ax.ylabelvisible = false)
        r < nrows && (ax.xticklabelsvisible = false; ax.xlabelvisible = false)
        r > 1 && (ax.titlevisible = false)
    end
    # Force equal column widths
    for c in 1:ncols
        Makie.colsize!(fig.layout, c, Makie.Relative(1 / ncols))
    end
    Makie.colgap!(fig.layout, colgap)
    Makie.rowgap!(fig.layout, rowgap)
end

# ============================================================================
# irfplot! — single-panel mutating methods for all IRF result types
# ============================================================================

# ---- IRFResult (point-identified, bootstrap / analytic bands) ---------------

function irfplot!(ax::Makie.Axis, irf::_IRFResult;
        var::Symbol, shock::Symbol,
        bandcolor = :steelblue, bandalpha = 0.2,
        linecolor = :black, linewidth = 2.0,
        drawzero = true, zerolinecolor = :gray60, zerolinestyle = :dash,
        xtickstep = 4, kwargs...)
    nms = _var_names(irf)
    vi = only(_resolve_indices(nms, [var], :var))
    si = only(_resolve_indices(nms, [shock], :shock))

    xvals = collect(0:MacroEconometricTools.horizon(irf))
    lb_all = lowerbounds(irf)
    ub_all = upperbounds(irf)
    cvgs = coverages(irf)

    lb_panel = [lb_all[k][vi, si, :] for k in eachindex(cvgs)]
    ub_panel = [ub_all[k][vi, si, :] for k in eachindex(cvgs)]
    y = irf.irf[vi, si, :]

    _plot_irf_panel!(ax, xvals, y, lb_panel, ub_panel, cvgs;
        drawzero, zerolinecolor, zerolinestyle,
        bandcolor, bandalpha, linecolor, linewidth, xtickstep)
    return ax
end

# ---- SignRestrictedIRFResult ------------------------------------------------

function irfplot!(ax::Makie.Axis, irf::_SignRestrictedIRFResult;
        var::Symbol, shock::Symbol,
        plot_type = :quantiles,
        bandcolor = :steelblue, bandalpha = 0.2,
        path_alpha = 0.02, path_color = :gray, path_linewidth = 0.5,
        linecolor = :black, linewidth = 2.0,
        drawzero = true, zerolinecolor = :gray60, zerolinestyle = :dash,
        xtickstep = 4, kwargs...)
    nms = _var_names(irf)
    vi = only(_resolve_indices(nms, [var], :var))
    si = only(_resolve_indices(nms, [shock], :shock))

    xvals = collect(0:MacroEconometricTools.horizon(irf))
    lb_all = lowerbounds(irf)
    ub_all = upperbounds(irf)
    cvgs = coverages(irf)

    Makie.xlims!(ax, xvals[1] - 0.4, xvals[end] + 0.4)
    if xtickstep > 0
        ax.xticks = xvals[1]:xtickstep:xvals[end]
    end

    if plot_type ∈ (:paths, :both)
        draws = irf.irf_draws[:, vi, si, :]
        _plot_paths!(ax, xvals, draws; path_alpha, path_color, path_linewidth)
    end

    if plot_type ∈ (:quantiles, :both) && !isempty(cvgs)
        lb_panel = [lb_all[k][vi, si, :] for k in eachindex(cvgs)]
        ub_panel = [ub_all[k][vi, si, :] for k in eachindex(cvgs)]
        _plot_irf_panel!(ax, xvals, irf.irf_median[vi, si, :],
            lb_panel, ub_panel, cvgs;
            drawzero, zerolinecolor, zerolinestyle,
            bandcolor, bandalpha, linecolor, linewidth, xtickstep)
    else
        Makie.lines!(ax, xvals, irf.irf_median[vi, si, :];
            color = linecolor, linewidth = linewidth)
        if drawzero
            Makie.hlines!(ax, [0.0]; color = zerolinecolor,
                linewidth = 1, linestyle = zerolinestyle)
        end
    end
    return ax
end

# ---- BayesianIRFResult (AxisArray-based) ------------------------------------

function irfplot!(ax::Makie.Axis, irf::_BayesianIRFResult;
        var::Symbol, shock::Symbol,
        plot_type = :quantiles,
        bandcolor = :steelblue, bandalpha = 0.2,
        path_alpha = 0.02, path_color = :gray, path_linewidth = 0.5,
        linecolor = :black, linewidth = 2.0,
        drawzero = true, zerolinecolor = :gray60, zerolinestyle = :dash,
        xtickstep = 4, kwargs...)
    var_axis = AxisArrays.axes(irf.data, Axis{:variable})
    shock_axis = AxisArrays.axes(irf.data, Axis{:shock})
    horizon_axis = AxisArrays.axes(irf.data, Axis{:horizon})

    all_vars = collect(AxisArrays.axisvalues(var_axis)[1])
    all_shocks = collect(AxisArrays.axisvalues(shock_axis)[1])
    horizons = collect(AxisArrays.axisvalues(horizon_axis)[1])

    vi = only(_resolve_indices_generic(all_vars, [var]))
    si = only(_resolve_indices_generic(all_shocks, [shock]))

    xvals = horizons
    lb_all = lowerbounds(irf)
    ub_all = upperbounds(irf)
    cvgs = coverages(irf)
    pt_est = point_estimate(irf)
    data_arr = Array(irf.data)

    Makie.xlims!(ax, xvals[1] - 0.4, xvals[end] + 0.4)
    if xtickstep > 0
        ax.xticks = xvals[1]:xtickstep:xvals[end]
    end

    if plot_type ∈ (:paths, :both)
        draws = data_arr[:, vi, si, :]
        _plot_paths!(ax, xvals, draws; path_alpha, path_color, path_linewidth)
    end

    lb_panel = [Array(lb_all[k])[vi, si, :] for k in eachindex(cvgs)]
    ub_panel = [Array(ub_all[k])[vi, si, :] for k in eachindex(cvgs)]
    y = Array(pt_est)[vi, si, :]

    if plot_type ∈ (:quantiles, :both)
        _plot_irf_panel!(ax, xvals, y, lb_panel, ub_panel, cvgs;
            drawzero, zerolinecolor, zerolinestyle,
            bandcolor, bandalpha, linecolor, linewidth, xtickstep)
    else
        Makie.lines!(ax, xvals, y;
            color = linecolor, linewidth = linewidth)
        if drawzero
            Makie.hlines!(ax, [0.0]; color = zerolinecolor,
                linewidth = 1, linestyle = zerolinestyle)
        end
    end
    return ax
end

# ---- LocalProjectionIRFResult (AxisArray-based) -----------------------------

function irfplot!(ax::Makie.Axis, irf::_LocalProjectionIRFResult;
        var::Symbol, shock::Symbol,
        bandcolor = :steelblue, bandalpha = 0.2,
        linecolor = :black, linewidth = 2.0,
        drawzero = true, zerolinecolor = :gray60, zerolinestyle = :dash,
        xtickstep = 4, kwargs...)
    response_axis = AxisArrays.axes(irf.data, Axis{:response})
    shock_axis = AxisArrays.axes(irf.data, Axis{:shock})
    horizon_axis = AxisArrays.axes(irf.data, Axis{:horizon})

    all_responses = collect(AxisArrays.axisvalues(response_axis)[1])
    all_shocks = collect(AxisArrays.axisvalues(shock_axis)[1])
    horizons = collect(AxisArrays.axisvalues(horizon_axis)[1])

    vi = only(_resolve_indices_generic(all_responses, [var]))
    si = only(_resolve_indices_generic(all_shocks, [shock]))

    xvals = horizons
    lb_all = lowerbounds(irf)
    ub_all = upperbounds(irf)
    cvgs = coverages(irf)
    pt_data = Array(irf.data)

    lb_panel = [Array(lb_all[k])[vi, si, :] for k in eachindex(cvgs)]
    ub_panel = [Array(ub_all[k])[vi, si, :] for k in eachindex(cvgs)]
    y = pt_data[vi, si, :]

    _plot_irf_panel!(ax, xvals, y, lb_panel, ub_panel, cvgs;
        drawzero, zerolinecolor, zerolinestyle,
        bandcolor, bandalpha, linecolor, linewidth, xtickstep)
    return ax
end

# ============================================================================
# irfplot — IRFResult  (point-identified, bootstrap / analytic bands)
# ============================================================================

function irfplot(irf::_IRFResult;
        # Variable / shock selection
        vars = :all, shocks = :all,
        pretty_vars = nothing, pretty_shocks = nothing,
        # Bands
        bandcolor = :steelblue, bandalpha = 0.2,
        # Line
        linecolor = :black, linewidth = 2.0,
        # Zero line
        drawzero = true, zerolinecolor = :gray60, zerolinestyle = :dash,
        # Ticks
        xtickstep = 4,
        # Layout
        figure = (;), size = nothing,
        title = nothing, title_fontsize = 20, title_font = :bold,
        linkxaxes = true, linkyaxes = :row,
        colgap = 10, rowgap = 10,
        kwargs...)
    setup = _prepare_irf_plot(irf;
        vars = vars, shocks = shocks,
        pretty_vars = pretty_vars, pretty_shocks = pretty_shocks)

    xvals = collect(0:MacroEconometricTools.horizon(irf))
    lb_all = lowerbounds(irf)
    ub_all = upperbounds(irf)
    cvgs = coverages(irf)
    nrows = length(setup.idxvars)
    ncols = length(setup.idxshocks)

    fig = _make_figure(; figure, size, title, title_fontsize, title_font)
    axes_matrix = Matrix{Makie.Axis}(undef, nrows, ncols)

    for (ri, vi) in enumerate(setup.idxvars), (ci, si) in enumerate(setup.idxshocks)

        ax = Makie.Axis(fig[ri, ci];
            title = setup.shock_labels[ci],
            ylabel = setup.var_labels[ri],
            xlabel = "Horizon")
        axes_matrix[ri, ci] = ax

        lb_panel = [lb_all[k][vi, si, :] for k in eachindex(cvgs)]
        ub_panel = [ub_all[k][vi, si, :] for k in eachindex(cvgs)]
        y = irf.irf[vi, si, :]

        _plot_irf_panel!(ax, xvals, y, lb_panel, ub_panel, cvgs;
            drawzero, zerolinecolor, zerolinestyle,
            bandcolor, bandalpha, linecolor, linewidth, xtickstep)
    end

    _configure_axes!(fig, axes_matrix, nrows, ncols;
        linkxaxes, linkyaxes, colgap, rowgap)
    return fig
end

# ============================================================================
# irfplot — SignRestrictedIRFResult
# ============================================================================

function irfplot(irf::_SignRestrictedIRFResult;
        vars = :all, shocks = :all,
        pretty_vars = nothing, pretty_shocks = nothing,
        # Plot type
        plot_type = :quantiles,  # :quantiles, :paths, :both
        # Bands
        bandcolor = :steelblue, bandalpha = 0.2,
        # Paths
        path_alpha = 0.02, path_color = :gray, path_linewidth = 0.5,
        # Line
        linecolor = :black, linewidth = 2.0,
        # Zero
        drawzero = true, zerolinecolor = :gray60, zerolinestyle = :dash,
        xtickstep = 4,
        # Layout
        figure = (;), size = nothing,
        title = nothing, title_fontsize = 20, title_font = :bold,
        linkxaxes = true, linkyaxes = :row,
        colgap = 10, rowgap = 10,
        kwargs...)
    setup = _prepare_irf_plot(irf;
        vars = vars, shocks = shocks,
        pretty_vars = pretty_vars, pretty_shocks = pretty_shocks)

    xvals = collect(0:MacroEconometricTools.horizon(irf))
    lb_all = lowerbounds(irf)
    ub_all = upperbounds(irf)
    cvgs = coverages(irf)
    nrows = length(setup.idxvars)
    ncols = length(setup.idxshocks)

    fig = _make_figure(; figure, size, title, title_fontsize, title_font)
    axes_matrix = Matrix{Makie.Axis}(undef, nrows, ncols)

    for (ri, vi) in enumerate(setup.idxvars), (ci, si) in enumerate(setup.idxshocks)

        ax = Makie.Axis(fig[ri, ci];
            title = setup.shock_labels[ci],
            ylabel = setup.var_labels[ri],
            xlabel = "Horizon")
        axes_matrix[ri, ci] = ax

        Makie.xlims!(ax, xvals[1] - 0.4, xvals[end] + 0.4)
        if xtickstep > 0
            ax.xticks = xvals[1]:xtickstep:xvals[end]
        end

        # Individual draws
        if plot_type ∈ (:paths, :both)
            draws = irf.irf_draws[:, vi, si, :]   # (n_draws, horizon)
            _plot_paths!(ax, xvals, draws; path_alpha, path_color, path_linewidth)
        end

        # Quantile bands
        if plot_type ∈ (:quantiles, :both) && !isempty(cvgs)
            lb_panel = [lb_all[k][vi, si, :] for k in eachindex(cvgs)]
            ub_panel = [ub_all[k][vi, si, :] for k in eachindex(cvgs)]
            _plot_irf_panel!(ax, xvals, irf.irf_median[vi, si, :],
                lb_panel, ub_panel, cvgs;
                drawzero, zerolinecolor, zerolinestyle,
                bandcolor, bandalpha, linecolor, linewidth, xtickstep)
        else
            # Median line only (paths mode without bands)
            Makie.lines!(ax, xvals, irf.irf_median[vi, si, :];
                color = linecolor, linewidth = linewidth)
            if drawzero
                Makie.hlines!(ax, [0.0]; color = zerolinecolor,
                    linewidth = 1, linestyle = zerolinestyle)
            end
        end
    end

    _configure_axes!(fig, axes_matrix, nrows, ncols;
        linkxaxes, linkyaxes, colgap, rowgap)
    return fig
end

# ============================================================================
# irfplot — BayesianIRFResult  (AxisArray-based)
# ============================================================================

function irfplot(irf::_BayesianIRFResult;
        vars = :all, shocks = :all,
        pretty_vars = nothing, pretty_shocks = nothing,
        plot_type = :quantiles,
        bandcolor = :steelblue, bandalpha = 0.2,
        path_alpha = 0.02, path_color = :gray, path_linewidth = 0.5,
        linecolor = :black, linewidth = 2.0,
        drawzero = true, zerolinecolor = :gray60, zerolinestyle = :dash,
        xtickstep = 4,
        figure = (;), size = nothing,
        title = nothing, title_fontsize = 20, title_font = :bold,
        linkxaxes = true, linkyaxes = :row,
        colgap = 10, rowgap = 10,
        kwargs...)
    var_axis = AxisArrays.axes(irf.data, Axis{:variable})
    shock_axis = AxisArrays.axes(irf.data, Axis{:shock})
    horizon_axis = AxisArrays.axes(irf.data, Axis{:horizon})

    all_vars = collect(AxisArrays.axisvalues(var_axis)[1])
    all_shocks = collect(AxisArrays.axisvalues(shock_axis)[1])
    horizons = collect(AxisArrays.axisvalues(horizon_axis)[1])

    idxvars = vars === :all ? collect(1:length(all_vars)) :
              _resolve_indices_generic(all_vars, vars)
    idxshocks = shocks === :all ? collect(1:length(all_shocks)) :
                _resolve_indices_generic(all_shocks, shocks)

    var_labels = pretty_vars === nothing ? string.(all_vars[idxvars]) : pretty_vars
    shock_labels = pretty_shocks === nothing ?
                   string.(all_shocks[idxshocks]) .* " shock" : pretty_shocks

    xvals = horizons
    lb_all = lowerbounds(irf)
    ub_all = upperbounds(irf)
    cvgs = coverages(irf)
    pt_est = point_estimate(irf)
    nrows = length(idxvars)
    ncols = length(idxshocks)

    fig = _make_figure(; figure, size, title, title_fontsize, title_font)
    axes_matrix = Matrix{Makie.Axis}(undef, nrows, ncols)

    data_arr = Array(irf.data)   # materialise once

    for (ri, vi) in enumerate(idxvars), (ci, si) in enumerate(idxshocks)

        ax = Makie.Axis(fig[ri, ci];
            title = shock_labels[ci],
            ylabel = var_labels[ri],
            xlabel = "Horizon")
        axes_matrix[ri, ci] = ax

        Makie.xlims!(ax, xvals[1] - 0.4, xvals[end] + 0.4)
        if xtickstep > 0
            ax.xticks = xvals[1]:xtickstep:xvals[end]
        end

        # Paths
        if plot_type ∈ (:paths, :both)
            draws = data_arr[:, vi, si, :]   # (n_draws, horizon)
            _plot_paths!(ax, xvals, draws; path_alpha, path_color, path_linewidth)
        end

        # Bands + median
        lb_panel = [Array(lb_all[k])[vi, si, :] for k in eachindex(cvgs)]
        ub_panel = [Array(ub_all[k])[vi, si, :] for k in eachindex(cvgs)]
        y = Array(pt_est)[vi, si, :]

        if plot_type ∈ (:quantiles, :both)
            _plot_irf_panel!(ax, xvals, y, lb_panel, ub_panel, cvgs;
                drawzero, zerolinecolor, zerolinestyle,
                bandcolor, bandalpha, linecolor, linewidth, xtickstep)
        else
            Makie.lines!(ax, xvals, y;
                color = linecolor, linewidth = linewidth)
            if drawzero
                Makie.hlines!(ax, [0.0]; color = zerolinecolor,
                    linewidth = 1, linestyle = zerolinestyle)
            end
        end
    end

    _configure_axes!(fig, axes_matrix, nrows, ncols;
        linkxaxes, linkyaxes, colgap, rowgap)
    return fig
end

# ============================================================================
# irfplot — LocalProjectionIRFResult  (AxisArray-based)
# ============================================================================

function irfplot(irf::_LocalProjectionIRFResult;
        vars = :all, shocks = :all,
        pretty_vars = nothing, pretty_shocks = nothing,
        bandcolor = :steelblue, bandalpha = 0.2,
        linecolor = :black, linewidth = 2.0,
        drawzero = true, zerolinecolor = :gray60, zerolinestyle = :dash,
        xtickstep = 4,
        figure = (;), size = nothing,
        title = nothing, title_fontsize = 20, title_font = :bold,
        linkxaxes = true, linkyaxes = :row,
        colgap = 10, rowgap = 10,
        kwargs...)
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

    xvals = horizons
    lb_all = lowerbounds(irf)
    ub_all = upperbounds(irf)
    cvgs = coverages(irf)
    pt_data = Array(irf.data)
    nrows = length(idxvars)
    ncols = length(idxshocks)

    fig = _make_figure(; figure, size, title, title_fontsize, title_font)
    axes_matrix = Matrix{Makie.Axis}(undef, nrows, ncols)

    for (ri, vi) in enumerate(idxvars), (ci, si) in enumerate(idxshocks)

        ax = Makie.Axis(fig[ri, ci];
            title = shock_labels[ci],
            ylabel = var_labels[ri],
            xlabel = "Horizon")
        axes_matrix[ri, ci] = ax

        lb_panel = [Array(lb_all[k])[vi, si, :] for k in eachindex(cvgs)]
        ub_panel = [Array(ub_all[k])[vi, si, :] for k in eachindex(cvgs)]
        y = pt_data[vi, si, :]

        _plot_irf_panel!(ax, xvals, y, lb_panel, ub_panel, cvgs;
            drawzero, zerolinecolor, zerolinestyle,
            bandcolor, bandalpha, linecolor, linewidth, xtickstep)
    end

    _configure_axes!(fig, axes_matrix, nrows, ncols;
        linkxaxes, linkyaxes, colgap, rowgap)
    return fig
end

# ============================================================================
# MCMC Diagnostic Recipes
# ============================================================================

Makie.@recipe(MCMCTrace, samples) do scene
    Makie.Theme(;
        linewidth = 1.5,
        alpha = 0.7,
        chain_colors = nothing
    )
end

function Makie.plot!(plot::MCMCTrace)
    samples = Makie.to_value(plot[1])
    attrs = plot.attributes
    lw = Makie.to_value(attrs[:linewidth])
    alpha = Makie.to_value(attrs[:alpha])

    if samples isa AbstractVector{<:AbstractVector}
        chain_colors = Makie.to_value(attrs[:chain_colors])
        if chain_colors === nothing
            chain_colors = Makie.wong_colors()
        end
        for (i, chain) in enumerate(samples)
            col = Makie.RGBAf(Makie.to_color(chain_colors[mod1(i, length(chain_colors))]), Float32(alpha))
            Makie.lines!(plot, 1:length(chain), collect(chain); color = col, linewidth = lw)
        end
    else
        col = Makie.RGBAf(Makie.to_color(:blue), Float32(alpha))
        Makie.lines!(plot, 1:length(samples), collect(samples); color = col, linewidth = lw)
    end
    return plot
end

Makie.@recipe(MCMCDensity, samples) do scene
    Makie.Theme(;
        linewidth = 2.0,
        combined_color = :black
    )
end

function Makie.plot!(plot::MCMCDensity)
    samples = Makie.to_value(plot[1])
    attrs = plot.attributes
    lw = Makie.to_value(attrs[:linewidth])
    combined_color = Makie.to_color(Makie.to_value(attrs[:combined_color]))
    all_samples = samples isa AbstractVector{<:AbstractVector} ? vcat(samples...) :
                  collect(samples)
    Makie.density!(plot, all_samples; color = combined_color, linewidth = lw)
    return plot
end

Makie.@recipe(MCMCHistogram, samples) do scene
    Makie.Theme(;
        bins = 50,
        show_kde = true,
        kde_linewidth = 2.0,
        color = :blue,
        kde_color = :black
    )
end

function Makie.plot!(plot::MCMCHistogram)
    samples = Makie.to_value(plot[1])
    attrs = plot.attributes
    bins = Makie.to_value(attrs[:bins])
    show_kde = Makie.to_value(attrs[:show_kde])
    kde_linewidth = Makie.to_value(attrs[:kde_linewidth])
    color = Makie.to_color(Makie.to_value(attrs[:color]))
    kde_color = Makie.to_color(Makie.to_value(attrs[:kde_color]))
    all_samples = samples isa AbstractVector{<:AbstractVector} ? vcat(samples...) :
                  collect(samples)
    Makie.hist!(plot, all_samples; bins = bins,
        color = Makie.RGBAf(color, 0.6f0), normalization = :pdf)
    if show_kde
        Makie.density!(plot, all_samples; color = kde_color, linewidth = kde_linewidth)
    end
    return plot
end

end # module
