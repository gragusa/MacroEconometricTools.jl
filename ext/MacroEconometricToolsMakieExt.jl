module MacroEconometricToolsMakieExt

using Makie: Makie
using MacroEconometricTools: MacroEconometricTools, AbstractIRFResult,
    point_estimate, lowerbounds, upperbounds, coverages, varnames, shocknames,
    get_draws, horizon

import MacroEconometricTools: irfplot, irfplot!

# Selection and label helpers shared with the Plots recipe (src/plots_recipes.jl)
const MET = MacroEconometricTools

# ============================================================================
# Shared panel-drawing core
# ============================================================================

"""
    _plot_irf_panel!(ax, xvals, y, lb, ub, cvgs; <keyword arguments>)

Draw a single IRF panel: confidence/credible bands (widest first), point
estimate line, and optional zero reference line.

`lb` and `ub` are vectors of vectors (one per coverage level), already sliced
to the correct variable/shock combination.  All scaling must be applied to the
data *before* calling this function (use `rescale`).
"""
function _plot_irf_panel!(
        ax, xvals, y, lb, ub, cvgs;
        drawzero = true, zerolinecolor = :gray60, zerolinestyle = :dash,
        bandcolor = :steelblue, bandalpha = 0.2,
        linecolor = :black, linewidth = 2.0,
        xtickstep = 4
    )
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
    Makie.lines!(
        ax, xvals, y;
        color = linecolor, linewidth = linewidth
    )

    # Zero reference
    return if drawzero
        Makie.hlines!(
            ax, [0.0]; color = zerolinecolor,
            linewidth = 1, linestyle = zerolinestyle
        )
    end
end

"""
    _plot_paths!(ax, xvals, draws; path_alpha, path_color, path_linewidth)

Overlay individual IRF draws as faint lines (sign-restricted / Bayesian).
"""
function _plot_paths!(
        ax, xvals, draws::AbstractMatrix;
        path_alpha = 0.02, path_color = :gray,
        path_linewidth = 0.5
    )
    col = Makie.RGBAf(Makie.to_color(path_color), Float32(path_alpha))
    for i in axes(draws, 1)
        Makie.lines!(
            ax, xvals, view(draws, i, :);
            color = col, linewidth = path_linewidth
        )
    end
    return
end

# ============================================================================
# Figure / layout helpers
# ============================================================================

function _make_figure(;
        figure = (;), size = nothing, title = nothing,
        title_fontsize = 20, title_font = :bold
    )
    fig_kw = Dict{Symbol, Any}(pairs(figure)...)
    if size !== nothing
        fig_kw[:size] = size
    end
    fig = Makie.Figure(; fig_kw...)
    if title !== nothing
        Makie.Label(
            fig[0, :]; text = title, fontsize = title_fontsize,
            font = title_font
        )
    end
    return fig
end

function _configure_axes!(
        fig, axes_matrix, nrows, ncols;
        linkxaxes = true, linkyaxes = :row,
        colgap = 10, rowgap = 10
    )
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
    return Makie.rowgap!(fig.layout, rowgap)
end

# ============================================================================
# Panel data: plain arrays materialized once per figure
# ============================================================================

# Point estimate, bands, and (when paths are requested) draws as plain arrays,
# indexed (variable, shock, horizon) and (draw, variable, shock, horizon).
function _panel_data(irf::AbstractIRFResult, plot_type)
    return (
        xvals = collect(0:horizon(irf)),
        point = Array(point_estimate(irf)),
        lower = [Array(l) for l in lowerbounds(irf)],
        upper = [Array(u) for u in upperbounds(irf)],
        coverages = coverages(irf),
        draws = plot_type === :quantiles ? nothing : Array(get_draws(irf)),
    )
end

function _draw_irf_panel!(
        ax, pd, vi, si, plot_type;
        bandcolor, bandalpha, path_alpha, path_color, path_linewidth,
        linecolor, linewidth, drawzero, zerolinecolor, zerolinestyle, xtickstep
    )
    xvals = pd.xvals
    y = pd.point[vi, si, :]
    Makie.xlims!(ax, xvals[1] - 0.4, xvals[end] + 0.4)
    if xtickstep > 0
        ax.xticks = xvals[1]:xtickstep:xvals[end]
    end

    if pd.draws !== nothing
        _plot_paths!(
            ax, xvals, pd.draws[:, vi, si, :];
            path_alpha, path_color, path_linewidth
        )
    end

    if plot_type !== :paths
        lb_panel = [l[vi, si, :] for l in pd.lower]
        ub_panel = [u[vi, si, :] for u in pd.upper]
        _plot_irf_panel!(
            ax, xvals, y, lb_panel, ub_panel, pd.coverages;
            drawzero, zerolinecolor, zerolinestyle,
            bandcolor, bandalpha, linecolor, linewidth, xtickstep
        )
    else
        Makie.lines!(ax, xvals, y; color = linecolor, linewidth = linewidth)
        if drawzero
            Makie.hlines!(
                ax, [0.0]; color = zerolinecolor,
                linewidth = 1, linestyle = zerolinestyle
            )
        end
    end
    return ax
end

# ============================================================================
# irfplot! — single panel
# ============================================================================

function irfplot!(
        ax::Makie.Axis, irf::AbstractIRFResult;
        var, shock,
        plot_type = :quantiles,
        bandcolor = :steelblue, bandalpha = 0.2,
        path_alpha = 0.02, path_color = :gray, path_linewidth = 0.5,
        linecolor = :black, linewidth = 2.0,
        drawzero = true, zerolinecolor = :gray60, zerolinestyle = :dash,
        xtickstep = 4, kwargs...
    )
    MET._check_plot_type(irf, plot_type)
    vi = only(MET._resolve_selection(varnames(irf), [var], :var))
    si = only(MET._resolve_selection(shocknames(irf), [shock], :shock))
    pd = _panel_data(irf, plot_type)
    _draw_irf_panel!(
        ax, pd, vi, si, plot_type;
        bandcolor, bandalpha, path_alpha, path_color, path_linewidth,
        linecolor, linewidth, drawzero, zerolinecolor, zerolinestyle, xtickstep
    )
    return ax
end

# ============================================================================
# irfplot — grid of panels
# ============================================================================

function irfplot(
        irf::AbstractIRFResult;
        # Variable / shock selection
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
        # Zero line
        drawzero = true, zerolinecolor = :gray60, zerolinestyle = :dash,
        # Ticks
        xtickstep = 4,
        # Layout
        figure = (;), size = nothing,
        title = nothing, title_fontsize = 20, title_font = :bold,
        linkxaxes = true, linkyaxes = :row,
        colgap = 10, rowgap = 10,
        kwargs...
    )
    MET._check_plot_type(irf, plot_type)
    setup = MET._prepare_irf_plot(irf; vars, shocks, pretty_vars, pretty_shocks)
    pd = _panel_data(irf, plot_type)
    nrows = length(setup.idxvars)
    ncols = length(setup.idxshocks)

    fig = _make_figure(; figure, size, title, title_fontsize, title_font)
    axes_matrix = Matrix{Makie.Axis}(undef, nrows, ncols)

    for (ri, vi) in enumerate(setup.idxvars), (ci, si) in enumerate(setup.idxshocks)

        ax = Makie.Axis(
            fig[ri, ci];
            title = setup.shock_labels[ci],
            ylabel = setup.var_labels[ri],
            xlabel = "Horizon"
        )
        axes_matrix[ri, ci] = ax
        _draw_irf_panel!(
            ax, pd, vi, si, plot_type;
            bandcolor, bandalpha, path_alpha, path_color, path_linewidth,
            linecolor, linewidth, drawzero, zerolinecolor, zerolinestyle, xtickstep
        )
    end

    _configure_axes!(
        fig, axes_matrix, nrows, ncols;
        linkxaxes, linkyaxes, colgap, rowgap
    )
    return fig
end

end # module
