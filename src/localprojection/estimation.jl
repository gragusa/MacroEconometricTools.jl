# ============================================================================
# Local Projection Estimation
# ============================================================================
# Placeholder for local projection methods

"""
    estimate(::Type{LocalProjection}, Y, n_lags; kwargs...)

Estimate local projection model (to be implemented).
"""
function estimate(spec::Type{LocalProjection}, Y::AbstractMatrix{T}, n_lags::Union{Int,Symbol};
                  kwargs...) where T<:AbstractFloat
    error("Local Projection estimation not yet implemented. Coming in next phase.")
end
