module MacroEconometricToolsMakieExt

using Makie
using MacroEconometricTools

const _IRFResult = MacroEconometricTools.IRFResult

# Helper functions
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

Makie.@recipe(IRFPlotMakie, irf::_IRFResult) do plot
    Makie.Attributes(
        vars = :all,
        shocks = :all,
        pretty_shocks = nothing,
        pretty_vars = nothing,
        drawzero = true,
        zerolinecolor = :gray70,
        bandcolor = :red,
        bandalpha = 0.25,
        linecolor = :black,
        linewidth = 2.0,
        xtickstep = 6,
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
    coverages = irf.coverage
    nrows = length(setup.idxvars)
    ncols = length(setup.idxshocks)

    for (row_idx, var_idx) in enumerate(setup.idxvars)
        for (col_idx, shock_idx) in enumerate(setup.idxshocks)
            ax = Makie.Axis(plot; row=row_idx, col=col_idx)
            ax.title[] = setup.shock_labels[col_idx]
            ax.ylabel[] = col_idx == 1 ? setup.var_labels[row_idx] : ""
            ax.xlabel[] = row_idx == nrows ? "Horizon" : ""
            ax.xlabelvisible[] = row_idx == nrows
            ax.ylabelvisible[] = col_idx == 1
            Makie.xlims!(ax, -0.2, xvals[end])
            if xtickstep > 0
                Makie.xticks!(ax, 0:xtickstep:xvals[end])
            end

            if !isempty(coverages) && !isempty(lb)
                for (cov_idx, _) in enumerate(coverages)
                    lower = lb[cov_idx][:, var_idx, shock_idx]
                    upper = ub[cov_idx][:, var_idx, shock_idx]
                    alpha = clamp(bandalpha / cov_idx, 0f0, 1f0)
                    color = RGBAf0(bandcolor, alpha)
                    Makie.band!(plot, xvals, lower, upper; axis=ax, color=color)
                end
            end

            y = irf.irf[:, var_idx, shock_idx]
            Makie.lines!(plot, xvals, y; axis=ax, color=linecolor, linewidth=linewidth)
            if drawzero
                Makie.hlines!(plot, [0.0]; axis=ax, color=zerolinecolor, linewidth=1, linestyle=:dash)
            end
        end
    end

    return plot
end

function Makie.irfplot(irf::_IRFResult; kwargs...)
    Makie.plot(IRFPlotMakie(irf; kwargs...))
end

end
