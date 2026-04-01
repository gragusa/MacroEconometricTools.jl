module MacroEconometricToolsMakieExt

using Makie
using MacroEconometricTools
using AxisArrays: AxisArrays, Axis

import MacroEconometricTools: irfplot

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
# Helper functions
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
# Helper to plot a single IRF panel into an axis
# ============================================================================

function _plot_irf_panel!(ax, xvals, y, lb, ub, cvgs;
        irf_scale = 1.0, drawzero = true, zerolinecolor = :gray70,
        bandcolor = :red, bandalpha = 0.25, linecolor = :black,
        linewidth = 2.0, xtickstep = 6)
    Makie.xlims!(ax, -0.2, xvals[end])
    if xtickstep > 0
        ax.xticks = 0:xtickstep:xvals[end]
    end

    # Confidence bands
    if !isempty(cvgs) && !isempty(lb)
        for (cov_idx, _) in enumerate(cvgs)
            lower = lb[cov_idx] .* irf_scale
            upper = ub[cov_idx] .* irf_scale
            alpha = clamp(bandalpha / cov_idx, 0.0f0, 1.0f0)
            color = Makie.RGBAf(Makie.to_color(bandcolor), alpha)
            Makie.band!(ax, xvals, lower, upper; color = color)
        end
    end

    # Point estimate line
    Makie.lines!(ax, xvals, y .* irf_scale;
        color = linecolor, linewidth = linewidth)

    # Zero line
    if drawzero
        Makie.hlines!(ax, [0.0]; color = zerolinecolor,
            linewidth = 1, linestyle = :dash)
    end
end

# ============================================================================
# irfplot for IRFResult
# ============================================================================

function irfplot(irf::_IRFResult;
        vars = :all, shocks = :all,
        pretty_shocks = nothing, pretty_vars = nothing,
        irf_scale = 1.0, drawzero = true, zerolinecolor = :gray70,
        bandcolor = :red, bandalpha = 0.25, linecolor = :black,
        linewidth = 2.0, xtickstep = 6,
        figure = (;), kwargs...)

    setup = _prepare_irf_plot(irf;
        vars = vars, shocks = shocks,
        pretty_vars = pretty_vars, pretty_shocks = pretty_shocks)

    xvals = collect(0:MacroEconometricTools.horizon(irf))
    lb_all = lowerbounds(irf)
    ub_all = upperbounds(irf)
    cvgs = coverages(irf)
    nrows = length(setup.idxvars)
    ncols = length(setup.idxshocks)

    fig = Makie.Figure(; figure...)

    for (row_idx, var_idx) in enumerate(setup.idxvars)
        for (col_idx, shock_idx) in enumerate(setup.idxshocks)
            ax = Makie.Axis(fig[row_idx, col_idx];
                title = setup.shock_labels[col_idx],
                ylabel = col_idx == 1 ? setup.var_labels[row_idx] : "",
                xlabel = row_idx == nrows ? "Horizon" : "",
                xlabelvisible = row_idx == nrows,
                ylabelvisible = col_idx == 1)

            lb_panel = [lb_all[ci][:, var_idx, shock_idx] for ci in eachindex(cvgs)]
            ub_panel = [ub_all[ci][:, var_idx, shock_idx] for ci in eachindex(cvgs)]
            y = irf.irf[:, var_idx, shock_idx]

            _plot_irf_panel!(ax, xvals, y, lb_panel, ub_panel, cvgs;
                irf_scale = irf_scale, drawzero = drawzero,
                zerolinecolor = zerolinecolor, bandcolor = bandcolor,
                bandalpha = bandalpha, linecolor = linecolor,
                linewidth = linewidth, xtickstep = xtickstep)
        end
    end

    return fig
end

# ============================================================================
# irfplot for SignRestrictedIRFResult
# ============================================================================

function irfplot(irf::_SignRestrictedIRFResult;
        vars = :all, shocks = :all,
        pretty_shocks = nothing, pretty_vars = nothing,
        irf_scale = 1.0, plot_type = :quantiles,
        drawzero = true, zerolinecolor = :gray70,
        bandcolor = :red, bandalpha = 0.25,
        path_alpha = 0.02, path_color = :gray,
        linecolor = :black, linewidth = 2.0, xtickstep = 6,
        figure = (;), kwargs...)

    setup = _prepare_irf_plot(irf;
        vars = vars, shocks = shocks,
        pretty_vars = pretty_vars, pretty_shocks = pretty_shocks)

    xvals = collect(0:MacroEconometricTools.horizon(irf))
    lb_all = lowerbounds(irf)
    ub_all = upperbounds(irf)
    cvgs = coverages(irf)
    nrows = length(setup.idxvars)
    ncols = length(setup.idxshocks)

    fig = Makie.Figure(; figure...)

    for (row_idx, var_idx) in enumerate(setup.idxvars)
        for (col_idx, shock_idx) in enumerate(setup.idxshocks)
            ax = Makie.Axis(fig[row_idx, col_idx];
                title = setup.shock_labels[col_idx],
                ylabel = col_idx == 1 ? setup.var_labels[row_idx] : "",
                xlabel = row_idx == nrows ? "Horizon" : "",
                xlabelvisible = row_idx == nrows,
                ylabelvisible = col_idx == 1)

            Makie.xlims!(ax, -0.2, xvals[end])
            if xtickstep > 0
                ax.xticks = 0:xtickstep:xvals[end]
            end

            # Draw paths if requested
            if plot_type ∈ [:paths, :both]
                n_drw = size(irf.irf_draws, 1)
                for draw_idx in 1:n_drw
                    y_path = irf.irf_draws[draw_idx, :, var_idx, shock_idx] .* irf_scale
                    Makie.lines!(ax, xvals, y_path;
                        color = Makie.RGBAf(Makie.to_color(path_color), Float32(path_alpha)),
                        linewidth = 0.5)
                end
            end

            # Draw quantile bands
            if plot_type ∈ [:quantiles, :both] && !isempty(cvgs) && !isempty(lb_all)
                for (cov_idx, _) in enumerate(cvgs)
                    lower = lb_all[cov_idx][:, var_idx, shock_idx] .* irf_scale
                    upper = ub_all[cov_idx][:, var_idx, shock_idx] .* irf_scale
                    alpha = clamp(bandalpha / cov_idx, 0.0f0, 1.0f0)
                    color = Makie.RGBAf(Makie.to_color(bandcolor), alpha)
                    Makie.band!(ax, xvals, lower, upper; color = color)
                end
            end

            # Median line
            y = irf.irf_median[:, var_idx, shock_idx] .* irf_scale
            Makie.lines!(ax, xvals, y; color = linecolor, linewidth = linewidth)

            if drawzero
                Makie.hlines!(ax, [0.0]; color = zerolinecolor,
                    linewidth = 1, linestyle = :dash)
            end
        end
    end

    return fig
end

# ============================================================================
# irfplot for BayesianIRFResult (AxisArray-based)
# ============================================================================

function irfplot(irf::_BayesianIRFResult;
        vars = :all, shocks = :all,
        pretty_shocks = nothing, pretty_vars = nothing,
        irf_scale = 1.0, plot_type = :quantiles,
        drawzero = true, zerolinecolor = :gray70,
        bandcolor = :red, bandalpha = 0.25,
        path_alpha = 0.02, path_color = :gray,
        linecolor = :black, linewidth = 2.0, xtickstep = 6,
        figure = (;), kwargs...)

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
    shock_labels = pretty_shocks === nothing ? string.(all_shocks[idxshocks]) .* " shock" :
                   pretty_shocks

    xvals = horizons
    lb_all = lowerbounds(irf)
    ub_all = upperbounds(irf)
    cvgs = coverages(irf)
    pt_est = point_estimate(irf)
    nrows = length(idxvars)
    ncols = length(idxshocks)

    fig = Makie.Figure(; figure...)

    for (row_idx, var_idx) in enumerate(idxvars)
        for (col_idx, shock_idx) in enumerate(idxshocks)
            ax = Makie.Axis(fig[row_idx, col_idx];
                title = shock_labels[col_idx],
                ylabel = col_idx == 1 ? var_labels[row_idx] : "",
                xlabel = row_idx == nrows ? "Horizon" : "",
                xlabelvisible = row_idx == nrows,
                ylabelvisible = col_idx == 1)

            Makie.xlims!(ax, -0.2, xvals[end])
            if xtickstep > 0
                ax.xticks = 0:xtickstep:xvals[end]
            end

            # Draw paths if requested
            if plot_type ∈ [:paths, :both]
                n_drw = n_draws(irf)
                data_arr = Array(irf.data)
                for draw_idx in 1:n_drw
                    y_path = data_arr[draw_idx, var_idx, shock_idx, :] .* irf_scale
                    Makie.lines!(ax, xvals, y_path;
                        color = Makie.RGBAf(Makie.to_color(path_color), Float32(path_alpha)),
                        linewidth = 0.5)
                end
            end

            # Draw quantile bands (widest first)
            if plot_type ∈ [:quantiles, :both] && !isempty(cvgs) && !isempty(lb_all)
                for (cov_idx, _) in enumerate(reverse(cvgs))
                    rev_idx = length(cvgs) - cov_idx + 1
                    lower = Array(lb_all[rev_idx])[var_idx, shock_idx, :] .* irf_scale
                    upper = Array(ub_all[rev_idx])[var_idx, shock_idx, :] .* irf_scale
                    alpha = clamp(bandalpha * 0.8^(cov_idx-1), 0.0f0, 1.0f0)
                    color = Makie.RGBAf(Makie.to_color(bandcolor), alpha)
                    Makie.band!(ax, xvals, lower, upper; color = color)
                end
            end

            # Median line
            y = Array(pt_est)[var_idx, shock_idx, :] .* irf_scale
            Makie.lines!(ax, xvals, y; color = linecolor, linewidth = linewidth)

            if drawzero
                Makie.hlines!(ax, [0.0]; color = zerolinecolor,
                    linewidth = 1, linestyle = :dash)
            end
        end
    end

    return fig
end

# ============================================================================
# irfplot for LocalProjectionIRFResult (AxisArray-based)
# ============================================================================

function irfplot(irf::_LocalProjectionIRFResult;
        irf_scale = 1.0, drawzero = true, zerolinecolor = :gray70,
        bandcolor = :blue, bandalpha = 0.25, linecolor = :black,
        linewidth = 2.0, xtickstep = 6,
        figure = (;), kwargs...)

    response_axis = AxisArrays.axes(irf.data, Axis{:response})
    shock_axis = AxisArrays.axes(irf.data, Axis{:shock})
    horizon_axis = AxisArrays.axes(irf.data, Axis{:horizon})

    responses = collect(AxisArrays.axisvalues(response_axis)[1])
    shocks = collect(AxisArrays.axisvalues(shock_axis)[1])
    horizons = collect(AxisArrays.axisvalues(horizon_axis)[1])

    xvals = horizons
    lb_all = lowerbounds(irf)
    ub_all = upperbounds(irf)
    cvgs = coverages(irf)
    pt_data = Array(irf.data)
    nrows = length(responses)
    ncols = length(shocks)

    fig = Makie.Figure(; figure...)

    for (row_idx, response) in enumerate(responses)
        for (col_idx, shock) in enumerate(shocks)
            ax = Makie.Axis(fig[row_idx, col_idx];
                title = string(shock) * " shock",
                ylabel = col_idx == 1 ? string(response) : "",
                xlabel = row_idx == nrows ? "Horizon" : "",
                xlabelvisible = row_idx == nrows,
                ylabelvisible = col_idx == 1)

            lb_panel = [Array(lb_all[ci])[row_idx, col_idx, :] for ci in eachindex(reverse(cvgs))]
            ub_panel = [Array(ub_all[ci])[row_idx, col_idx, :] for ci in eachindex(reverse(cvgs))]
            y = pt_data[row_idx, col_idx, :]

            _plot_irf_panel!(ax, xvals, y, lb_panel, ub_panel, cvgs;
                irf_scale = irf_scale, drawzero = drawzero,
                zerolinecolor = zerolinecolor, bandcolor = bandcolor,
                bandalpha = bandalpha, linecolor = linecolor,
                linewidth = linewidth, xtickstep = xtickstep)
        end
    end

    return fig
end

# ============================================================================
# MCMC Diagnostic Recipes
# ============================================================================

Makie.@recipe(MCMCTrace, samples) do scene
    Makie.Theme(;
        linewidth = 1.5,
        alpha = 0.7,
        chain_colors = nothing,
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
            color = Makie.RGBAf(Makie.to_color(chain_colors[mod1(i, length(chain_colors))]), Float32(alpha))
            Makie.lines!(plot, 1:length(chain), collect(chain); color = color, linewidth = lw)
        end
    else
        color = Makie.RGBAf(Makie.to_color(:blue), Float32(alpha))
        Makie.lines!(plot, 1:length(samples), collect(samples); color = color, linewidth = lw)
    end

    return plot
end

Makie.@recipe(MCMCDensity, samples) do scene
    Makie.Theme(;
        linewidth = 2.0,
        combined_color = :black,
    )
end

function Makie.plot!(plot::MCMCDensity)
    samples = Makie.to_value(plot[1])
    attrs = plot.attributes

    lw = Makie.to_value(attrs[:linewidth])
    combined_color = Makie.to_color(Makie.to_value(attrs[:combined_color]))

    all_samples = samples isa AbstractVector{<:AbstractVector} ? vcat(samples...) : collect(samples)

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

    all_samples = samples isa AbstractVector{<:AbstractVector} ? vcat(samples...) : collect(samples)

    Makie.hist!(plot, all_samples; bins = bins,
        color = Makie.RGBAf(color, 0.6f0), normalization = :pdf)

    if show_kde
        Makie.density!(plot, all_samples; color = kde_color, linewidth = kde_linewidth)
    end

    return plot
end

end # module
