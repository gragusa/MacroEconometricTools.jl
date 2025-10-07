# ============================================================================
# IV-SVAR Estimation
# ============================================================================
# Placeholder for IV-SVAR estimation methods

"""
    estimate(::Type{IVSVAR}, Y, n_lags; instrument, kwargs...)

Estimate IV-SVAR model (to be implemented).
"""
function estimate(spec::Type{IVSVAR}, Y::AbstractMatrix{T}, n_lags::Int;
                  instrument::AbstractInstrument,
                  kwargs...) where T<:AbstractFloat
    error("IV-SVAR estimation not yet implemented. Coming in next phase.")
end
