module MacroEconometricToolsMakieExt

using Makie
using MacroEconometricTools
using AxisArrays: AxisArrays, Axis

const _IRFResult = MacroEconometricTools.IRFResult
const _SignRestrictedIRFResult = MacroEconometricTools.SignRestrictedIRFResult
const _BayesianIRFResult = MacroEconometricTools.BayesianIRFResult
const _LocalProjectionIRFResult = MacroEconometricTools.LocalProjectionIRFResult
const _AbstractIRFResult = MacroEconometricTools.AbstractIRFResult

# Use accessors from MacroEconometricTools
const lowerbounds = MacroEconometricTools.lowerbounds
const upperbounds = MacroEconometricTools.upperbounds
const coverages = MacroEconometricTools.coverages
const point_estimate = MacroEconometricTools.point_estimate
const n_draws = MacroEconometricTools.n_draws

# ============================================================================
# Helper functions for IRFResult (non-AxisArray based)
# ============================================================================

function _var_names(irf::_IRFResult)
    if haskey(irf.metadata, :names)
        names = irf.metadata.names
        return Symbol.(names)
    else
        return [Symbol("Y_$i") for i in 1:MacroEconometricTools.n_vars(irf)]
    end
end

function _var_names(irf::_SignRestrictedIRFResult)
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

function _prepare_irf_plot(
        irf::Union{_IRFResult, _SignRestrictedIRFResult}; vars = :all, shocks = :all,
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

# ============================================================================
# IRFPlotMakie Recipe for IRFResult
# ============================================================================

Makie.@recipe(IRFPlotMakie, irf::_IRFResult) do plot
    Makie.Attributes(
        vars = :all,
        shocks = :all,
        pretty_shocks = nothing,
        pretty_vars = nothing,
        irf_scale = 1.0,
        drawzero = true,
        zerolinecolor = :gray70,
        bandcolor = :red,
        bandalpha = 0.25,
        linecolor = :black,
        linewidth = 2.0,
        xtickstep = 6
    )
end

function Makie.plot!(plot::IRFPlotMakie)
    irf = Makie.to_value(plot[1])
    attrs = plot.attributes

    setup = _prepare_irf_plot(irf;
        vars = Makie.to_value(attrs[:vars]),
        shocks = Makie.to_value(attrs[:shocks]),
        pretty_vars = Makie.to_value(attrs[:pretty_vars]),
        pretty_shocks = Makie.to_value(attrs[:pretty_shocks])
    )

    irf_scale = Makie.to_value(attrs[:irf_scale])
    drawzero = Makie.to_value(attrs[:drawzero])
    zerolinecolor = Makie.to_color(Makie.to_value(attrs[:zerolinecolor]))
    bandcolor = Makie.to_color(Makie.to_value(attrs[:bandcolor]))
    bandalpha = Makie.to_value(attrs[:bandalpha])
    linecolor = Makie.to_color(Makie.to_value(attrs[:linecolor]))
    linewidth = Makie.to_value(attrs[:linewidth])
    xtickstep = Makie.to_value(attrs[:xtickstep])

    xvals = collect(0:MacroEconometricTools.horizon(irf))
    lb = lowerbounds(irf)
    ub = upperbounds(irf)
    cvgs = coverages(irf)
    nrows = length(setup.idxvars)
    ncols = length(setup.idxshocks)

    for (row_idx, var_idx) in enumerate(setup.idxvars)
        for (col_idx, shock_idx) in enumerate(setup.idxshocks)
            ax = Makie.Axis(plot; row = row_idx, col = col_idx)
            ax.title[] = setup.shock_labels[col_idx]
            ax.ylabel[] = col_idx == 1 ? setup.var_labels[row_idx] : ""
            ax.xlabel[] = row_idx == nrows ? "Horizon" : ""
            ax.xlabelvisible[] = row_idx == nrows
            ax.ylabelvisible[] = col_idx == 1
            Makie.xlims!(ax, -0.2, xvals[end])
            if xtickstep > 0
                Makie.xticks!(ax, 0:xtickstep:xvals[end])
            end

            if !isempty(cvgs) && !isempty(lb)
                for (cov_idx, _) in enumerate(cvgs)
                    lower = lb[cov_idx][:, var_idx, shock_idx] .* irf_scale
                    upper = ub[cov_idx][:, var_idx, shock_idx] .* irf_scale
                    alpha = clamp(bandalpha / cov_idx, 0.0f0, 1.0f0)
                    color = RGBAf0(bandcolor, alpha)
                    Makie.band!(plot, xvals, lower, upper; axis = ax, color = color)
                end
            end

            y = irf.irf[:, var_idx, shock_idx] .* irf_scale
            Makie.lines!(
                plot, xvals, y; axis = ax, color = linecolor, linewidth = linewidth)
            if drawzero
                Makie.hlines!(plot, [0.0]; axis = ax, color = zerolinecolor,
                    linewidth = 1, linestyle = :dash)
            end
        end
    end

    return plot
end

function Makie.irfplot(irf::_IRFResult; kwargs...)
    Makie.plot(IRFPlotMakie(irf; kwargs...))
end

# ============================================================================
# SignRestrictedIRFPlotMakie Recipe
# ============================================================================

Makie.@recipe(SignRestrictedIRFPlotMakie, irf::_SignRestrictedIRFResult) do plot
    Makie.Attributes(
        vars = :all,
        shocks = :all,
        pretty_shocks = nothing,
        pretty_vars = nothing,
        irf_scale = 1.0,
        plot_type = :quantiles,
        drawzero = true,
        zerolinecolor = :gray70,
        bandcolor = :red,
        bandalpha = 0.25,
        path_alpha = 0.02,
        path_color = :gray,
        linecolor = :black,
        linewidth = 2.0,
        xtickstep = 6
    )
end

function Makie.plot!(plot::SignRestrictedIRFPlotMakie)
    irf = Makie.to_value(plot[1])
    attrs = plot.attributes

    setup = _prepare_irf_plot(irf;
        vars = Makie.to_value(attrs[:vars]),
        shocks = Makie.to_value(attrs[:shocks]),
        pretty_vars = Makie.to_value(attrs[:pretty_vars]),
        pretty_shocks = Makie.to_value(attrs[:pretty_shocks])
    )

    irf_scale = Makie.to_value(attrs[:irf_scale])
    plot_type = Makie.to_value(attrs[:plot_type])
    drawzero = Makie.to_value(attrs[:drawzero])
    zerolinecolor = Makie.to_color(Makie.to_value(attrs[:zerolinecolor]))
    bandcolor = Makie.to_color(Makie.to_value(attrs[:bandcolor]))
    bandalpha = Makie.to_value(attrs[:bandalpha])
    path_alpha = Makie.to_value(attrs[:path_alpha])
    path_color = Makie.to_color(Makie.to_value(attrs[:path_color]))
    linecolor = Makie.to_color(Makie.to_value(attrs[:linecolor]))
    linewidth = Makie.to_value(attrs[:linewidth])
    xtickstep = Makie.to_value(attrs[:xtickstep])

    xvals = collect(0:MacroEconometricTools.horizon(irf))
    lb = lowerbounds(irf)
    ub = upperbounds(irf)
    cvgs = coverages(irf)
    nrows = length(setup.idxvars)
    ncols = length(setup.idxshocks)

    for (row_idx, var_idx) in enumerate(setup.idxvars)
        for (col_idx, shock_idx) in enumerate(setup.idxshocks)
            ax = Makie.Axis(plot; row = row_idx, col = col_idx)
            ax.title[] = setup.shock_labels[col_idx]
            ax.ylabel[] = col_idx == 1 ? setup.var_labels[row_idx] : ""
            ax.xlabel[] = row_idx == nrows ? "Horizon" : ""
            ax.xlabelvisible[] = row_idx == nrows
            ax.ylabelvisible[] = col_idx == 1
            Makie.xlims!(ax, -0.2, xvals[end])
            if xtickstep > 0
                Makie.xticks!(ax, 0:xtickstep:xvals[end])
            end

            # Draw paths if requested
            if plot_type ∈ [:paths, :both]
                n_drw = size(irf.irf_draws, 1)
                for draw_idx in 1:n_drw
                    y_path = irf.irf_draws[draw_idx, :, var_idx, shock_idx] .* irf_scale
                    Makie.lines!(plot, xvals, y_path; axis = ax,
                        color = RGBAf0(path_color, path_alpha), linewidth = 0.5)
                end
            end

            # Draw quantile bands
            if plot_type ∈ [:quantiles, :both] && !isempty(cvgs) && !isempty(lb)
                for (cov_idx, _) in enumerate(cvgs)
                    lower = lb[cov_idx][:, var_idx, shock_idx] .* irf_scale
                    upper = ub[cov_idx][:, var_idx, shock_idx] .* irf_scale
                    alpha = clamp(bandalpha / cov_idx, 0.0f0, 1.0f0)
                    color = RGBAf0(bandcolor, alpha)
                    Makie.band!(plot, xvals, lower, upper; axis = ax, color = color)
                end
            end

            # Median line
            y = irf.irf_median[:, var_idx, shock_idx] .* irf_scale
            Makie.lines!(
                plot, xvals, y; axis = ax, color = linecolor, linewidth = linewidth)

            if drawzero
                Makie.hlines!(plot, [0.0]; axis = ax, color = zerolinecolor,
                    linewidth = 1, linestyle = :dash)
            end
        end
    end

    return plot
end

# ============================================================================
# BayesianIRFPlotMakie Recipe (AxisArray-based)
# ============================================================================

Makie.@recipe(BayesianIRFPlotMakie, irf::_BayesianIRFResult) do plot
    Makie.Attributes(
        vars = :all,
        shocks = :all,
        pretty_shocks = nothing,
        pretty_vars = nothing,
        irf_scale = 1.0,
        plot_type = :quantiles,
        drawzero = true,
        zerolinecolor = :gray70,
        bandcolor = :red,
        bandalpha = 0.25,
        path_alpha = 0.02,
        path_color = :gray,
        linecolor = :black,
        linewidth = 2.0,
        xtickstep = 6
    )
end

function Makie.plot!(plot::BayesianIRFPlotMakie)
    irf = Makie.to_value(plot[1])
    attrs = plot.attributes

    # Get variable and shock names from AxisArray axes
    var_axis = AxisArrays.axes(irf.data, Axis{:variable})
    shock_axis = AxisArrays.axes(irf.data, Axis{:shock})
    horizon_axis = AxisArrays.axes(irf.data, Axis{:horizon})

    all_vars = collect(AxisArrays.axisvalues(var_axis)[1])
    all_shocks = collect(AxisArrays.axisvalues(shock_axis)[1])
    horizons = collect(AxisArrays.axisvalues(horizon_axis)[1])

    vars_sel = Makie.to_value(attrs[:vars])
    shocks_sel = Makie.to_value(attrs[:shocks])

    idxvars = vars_sel === :all ? collect(1:length(all_vars)) :
              _resolve_indices_generic(all_vars, vars_sel)
    idxshocks = shocks_sel === :all ? collect(1:length(all_shocks)) :
                _resolve_indices_generic(all_shocks, shocks_sel)

    pretty_vars = Makie.to_value(attrs[:pretty_vars])
    pretty_shocks = Makie.to_value(attrs[:pretty_shocks])
    var_labels = pretty_vars === nothing ? string.(all_vars[idxvars]) : pretty_vars
    shock_labels = pretty_shocks === nothing ? string.(all_shocks[idxshocks]) .* " shock" :
                   pretty_shocks

    irf_scale = Makie.to_value(attrs[:irf_scale])
    plot_type = Makie.to_value(attrs[:plot_type])
    drawzero = Makie.to_value(attrs[:drawzero])
    zerolinecolor = Makie.to_color(Makie.to_value(attrs[:zerolinecolor]))
    bandcolor = Makie.to_color(Makie.to_value(attrs[:bandcolor]))
    bandalpha = Makie.to_value(attrs[:bandalpha])
    path_alpha = Makie.to_value(attrs[:path_alpha])
    path_color = Makie.to_color(Makie.to_value(attrs[:path_color]))
    linecolor = Makie.to_color(Makie.to_value(attrs[:linecolor]))
    linewidth = Makie.to_value(attrs[:linewidth])
    xtickstep = Makie.to_value(attrs[:xtickstep])

    xvals = horizons
    lb = lowerbounds(irf)
    ub = upperbounds(irf)
    cvgs = coverages(irf)
    pt_est = point_estimate(irf)
    nrows = length(idxvars)
    ncols = length(idxshocks)

    for (row_idx, var_idx) in enumerate(idxvars)
        for (col_idx, shock_idx) in enumerate(idxshocks)
            ax = Makie.Axis(plot; row = row_idx, col = col_idx)
            ax.title[] = shock_labels[col_idx]
            ax.ylabel[] = col_idx == 1 ? var_labels[row_idx] : ""
            ax.xlabel[] = row_idx == nrows ? "Horizon" : ""
            ax.xlabelvisible[] = row_idx == nrows
            ax.ylabelvisible[] = col_idx == 1
            Makie.xlims!(ax, -0.2, xvals[end])
            if xtickstep > 0
                Makie.xticks!(ax, 0:xtickstep:xvals[end])
            end

            # Draw paths if requested
            if plot_type ∈ [:paths, :both]
                n_drw = n_draws(irf)
                data_arr = Array(irf.data)
                for draw_idx in 1:n_drw
                    y_path = data_arr[draw_idx, var_idx, shock_idx, :] .* irf_scale
                    Makie.lines!(plot, xvals, y_path; axis = ax,
                        color = RGBAf0(path_color, path_alpha), linewidth = 0.5)
                end
            end

            # Draw quantile bands (widest first)
            if plot_type ∈ [:quantiles, :both] && !isempty(cvgs) && !isempty(lb)
                for (cov_idx, _) in enumerate(reverse(cvgs))
                    rev_idx = length(cvgs) - cov_idx + 1
                    lower = Array(lb[rev_idx])[var_idx, shock_idx, :] .* irf_scale
                    upper = Array(ub[rev_idx])[var_idx, shock_idx, :] .* irf_scale
                    alpha = clamp(bandalpha * 0.8^(cov_idx-1), 0.0f0, 1.0f0)
                    color = RGBAf0(bandcolor, alpha)
                    Makie.band!(plot, xvals, lower, upper; axis = ax, color = color)
                end
            end

            # Median line
            y = Array(pt_est)[var_idx, shock_idx, :] .* irf_scale
            Makie.lines!(
                plot, xvals, y; axis = ax, color = linecolor, linewidth = linewidth)

            if drawzero
                Makie.hlines!(plot, [0.0]; axis = ax, color = zerolinecolor,
                    linewidth = 1, linestyle = :dash)
            end
        end
    end

    return plot
end

# ============================================================================
# LocalProjectionIRFPlotMakie Recipe (AxisArray-based)
# ============================================================================

Makie.@recipe(LocalProjectionIRFPlotMakie, irf::_LocalProjectionIRFResult) do plot
    Makie.Attributes(
        irf_scale = 1.0,
        drawzero = true,
        zerolinecolor = :gray70,
        bandcolor = :blue,
        bandalpha = 0.25,
        linecolor = :black,
        linewidth = 2.0,
        xtickstep = 6
    )
end

function Makie.plot!(plot::LocalProjectionIRFPlotMakie)
    irf = Makie.to_value(plot[1])
    attrs = plot.attributes

    # Get dimensions from AxisArray axes
    response_axis = AxisArrays.axes(irf.data, Axis{:response})
    shock_axis = AxisArrays.axes(irf.data, Axis{:shock})
    horizon_axis = AxisArrays.axes(irf.data, Axis{:horizon})

    responses = collect(AxisArrays.axisvalues(response_axis)[1])
    shocks = collect(AxisArrays.axisvalues(shock_axis)[1])
    horizons = collect(AxisArrays.axisvalues(horizon_axis)[1])

    irf_scale = Makie.to_value(attrs[:irf_scale])
    drawzero = Makie.to_value(attrs[:drawzero])
    zerolinecolor = Makie.to_color(Makie.to_value(attrs[:zerolinecolor]))
    bandcolor = Makie.to_color(Makie.to_value(attrs[:bandcolor]))
    bandalpha = Makie.to_value(attrs[:bandalpha])
    linecolor = Makie.to_color(Makie.to_value(attrs[:linecolor]))
    linewidth = Makie.to_value(attrs[:linewidth])
    xtickstep = Makie.to_value(attrs[:xtickstep])

    xvals = horizons
    lb = lowerbounds(irf)
    ub = upperbounds(irf)
    cvgs = coverages(irf)
    pt_data = Array(irf.data)
    nrows = length(responses)
    ncols = length(shocks)

    for (row_idx, response) in enumerate(responses)
        for (col_idx, shock) in enumerate(shocks)
            ax = Makie.Axis(plot; row = row_idx, col = col_idx)
            ax.title[] = string(shock) * " shock"
            ax.ylabel[] = col_idx == 1 ? string(response) : ""
            ax.xlabel[] = row_idx == nrows ? "Horizon" : ""
            ax.xlabelvisible[] = row_idx == nrows
            ax.ylabelvisible[] = col_idx == 1
            Makie.xlims!(ax, -0.2, xvals[end])
            if xtickstep > 0
                Makie.xticks!(ax, 0:xtickstep:xvals[end])
            end

            # Draw confidence bands (widest first)
            if !isempty(cvgs) && !isempty(lb)
                for (cov_idx, _) in enumerate(reverse(cvgs))
                    rev_idx = length(cvgs) - cov_idx + 1
                    lower = Array(lb[rev_idx])[row_idx, col_idx, :] .* irf_scale
                    upper = Array(ub[rev_idx])[row_idx, col_idx, :] .* irf_scale
                    alpha = clamp(bandalpha * 0.8^(cov_idx-1), 0.0f0, 1.0f0)
                    color = RGBAf0(bandcolor, alpha)
                    Makie.band!(plot, xvals, lower, upper; axis = ax, color = color)
                end
            end

            # Point estimate line
            y = pt_data[row_idx, col_idx, :] .* irf_scale
            Makie.lines!(
                plot, xvals, y; axis = ax, color = linecolor, linewidth = linewidth)

            if drawzero
                Makie.hlines!(plot, [0.0]; axis = ax, color = zerolinecolor,
                    linewidth = 1, linestyle = :dash)
            end
        end
    end

    return plot
end

# ============================================================================
# MCMC Diagnostic Plots (from BayesianVAR integration)
# ============================================================================

"""
    mcmctrace(samples; kwargs...)

Plot MCMC trace for posterior samples.

# Arguments
- `samples::AbstractVector`: Vector of MCMC samples

# Keyword Arguments
- `linewidth=1.5`: Line width
- `alpha=0.7`: Line alpha
- `chain_colors=nothing`: Colors for multiple chains
- `show_legend=true`: Whether to show legend
"""
Makie.@recipe(MCMCTrace, samples) do plot
    Makie.Attributes(
        linewidth = 1.5,
        alpha = 0.7,
        chain_colors = nothing,
        show_legend = true
    )
end

function Makie.plot!(plot::MCMCTrace)
    samples = Makie.to_value(plot[1])
    attrs = plot.attributes

    linewidth = Makie.to_value(attrs[:linewidth])
    alpha = Makie.to_value(attrs[:alpha])

    ax = Makie.Axis(plot)
    ax.xlabel[] = "Iteration"
    ax.ylabel[] = "Value"

    # Handle single chain or multiple chains
    if samples isa AbstractVector{<:AbstractVector}
        # Multiple chains
        chain_colors = Makie.to_value(attrs[:chain_colors])
        if chain_colors === nothing
            chain_colors = Makie.wong_colors()
        end
        for (i, chain) in enumerate(samples)
            color = RGBAf0(Makie.to_color(chain_colors[mod1(i, length(chain_colors))]), alpha)
            Makie.lines!(plot, 1:length(chain), chain; axis = ax,
                color = color, linewidth = linewidth,
                label = "Chain $i")
        end
        if Makie.to_value(attrs[:show_legend])
            Makie.axislegend(ax)
        end
    else
        # Single chain
        color = RGBAf0(Makie.to_color(:blue), alpha)
        Makie.lines!(plot, 1:length(samples), samples; axis = ax,
            color = color, linewidth = linewidth)
    end

    return plot
end

"""
    mcmcdensity(samples; kwargs...)

Plot posterior density estimate.

# Keyword Arguments
- `linewidth=2.0`: Line width
- `bandwidth=nothing`: KDE bandwidth (auto if nothing)
- `show_prior=true`: Whether to show prior (if provided)
- `prior_color=:red`: Prior line color
- `prior_linestyle=:dash`: Prior line style
"""
Makie.@recipe(MCMCDensity, samples) do plot
    Makie.Attributes(
        linewidth = 2.0,
        bandwidth = nothing,
        show_chains = true,
        show_combined = true,
        combined_color = :black,
        chain_colors = nothing,
        prior = nothing,
        prior_color = :red,
        prior_linestyle = :dash
    )
end

function Makie.plot!(plot::MCMCDensity)
    samples = Makie.to_value(plot[1])
    attrs = plot.attributes

    linewidth = Makie.to_value(attrs[:linewidth])
    combined_color = Makie.to_color(Makie.to_value(attrs[:combined_color]))

    ax = Makie.Axis(plot)
    ax.xlabel[] = "Value"
    ax.ylabel[] = "Density"

    # Combine all samples for density estimation
    all_samples = samples isa AbstractVector{<:AbstractVector} ? vcat(samples...) : samples

    # Simple kernel density estimate using Makie's density
    Makie.density!(
        plot, all_samples; axis = ax, color = combined_color, linewidth = linewidth)

    # Show prior if provided
    prior = Makie.to_value(attrs[:prior])
    if prior !== nothing
        prior_color = Makie.to_color(Makie.to_value(attrs[:prior_color]))
        prior_linestyle = Makie.to_value(attrs[:prior_linestyle])
        xrange = range(minimum(all_samples), maximum(all_samples), length = 200)
        # Assume prior has a pdf method
        if hasmethod(pdf, Tuple{typeof(prior), eltype(xrange)})
            prior_y = [pdf(prior, x) for x in xrange]
            Makie.lines!(plot, xrange, prior_y; axis = ax, color = prior_color,
                linewidth = linewidth, linestyle = prior_linestyle, label = "Prior")
        end
    end

    return plot
end

"""
    mcmchistogram(samples; kwargs...)

Plot histogram of posterior samples with optional KDE overlay.

# Keyword Arguments
- `bins=50`: Number of bins
- `show_kde=true`: Whether to show KDE overlay
- `kde_linewidth=2.0`: KDE line width
"""
Makie.@recipe(MCMCHistogram, samples) do plot
    Makie.Attributes(
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

    ax = Makie.Axis(plot)
    ax.xlabel[] = "Value"
    ax.ylabel[] = "Frequency"

    # Combine samples if multiple chains
    all_samples = samples isa AbstractVector{<:AbstractVector} ? vcat(samples...) : samples

    # Histogram
    Makie.hist!(plot, all_samples; axis = ax, bins = bins,
        color = RGBAf0(color, 0.6), normalization = :pdf)

    # KDE overlay
    if show_kde
        Makie.density!(
            plot, all_samples; axis = ax, color = kde_color, linewidth = kde_linewidth)
    end

    return plot
end

# ============================================================================
# Helper functions
# ============================================================================

function _resolve_indices_generic(all_names, selection)
    if selection isa AbstractVector{Symbol}
        return [findfirst(==(s), all_names) for s in selection]
    elseif selection isa AbstractVector{<:Integer}
        return selection
    else
        error("Selection must be :all, a vector of Symbols, or a vector of integers")
    end
end

# Convenience functions
function Makie.irfplot(irf::_SignRestrictedIRFResult; kwargs...)
    Makie.plot(SignRestrictedIRFPlotMakie(irf; kwargs...))
end

function Makie.irfplot(irf::_BayesianIRFResult; kwargs...)
    Makie.plot(BayesianIRFPlotMakie(irf; kwargs...))
end

function Makie.irfplot(irf::_LocalProjectionIRFResult; kwargs...)
    Makie.plot(LocalProjectionIRFPlotMakie(irf; kwargs...))
end

end # module
