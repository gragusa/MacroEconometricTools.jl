# ============================================================================
# IV-SVAR Estimation
# ============================================================================
# Placeholder for IV-SVAR estimation methods

"""
    StatsBase.fit(::Type{IVSVAR}, Y, n_lags; instrument, kwargs...)

Fit IV-SVAR model (to be implemented).

This follows the StatsBase.jl convention for model fitting.
"""
function StatsBase.fit(spec::Type{IVSVAR}, Y::AbstractMatrix{T}, n_lags::Int;
                       instrument::AbstractInstrument,
                       kwargs...) where T<:AbstractFloat
    error("IV-SVAR estimation not yet implemented. Coming in next phase.")
end
