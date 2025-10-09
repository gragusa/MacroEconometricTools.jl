# ============================================================================
# Local Projection Estimation
# ============================================================================
# Placeholder for local projection methods

"""
    StatsBase.fit(::Type{LocalProjection}, Y, n_lags; kwargs...)

Fit local projection model (to be implemented).

This follows the StatsBase.jl convention for model fitting.
"""
function StatsBase.fit(spec::Type{LocalProjection}, Y::AbstractMatrix{T}, n_lags::Union{Int,Symbol};
                       kwargs...) where T<:AbstractFloat
    error("Local Projection estimation not yet implemented. Coming in next phase.")
end
