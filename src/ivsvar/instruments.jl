# ============================================================================
# IV-SVAR Instrument Utilities
# ============================================================================
# Types are defined in types.jl (ExternalInstrument, ProxyIV)

# --- ExternalInstrument constructors ------------------------------------------

# Keyword-only: matrix input (default target_shock=1)
function ExternalInstrument(Z::AbstractMatrix{T};
        target_shock::Union{Int, Symbol} = 1,
        method::Symbol = :tsls) where {T <: AbstractFloat}
    return ExternalInstrument(Matrix{T}(Z), target_shock; method = method)
end

# Keyword-only: vector input (auto-reshape to T×1)
function ExternalInstrument(Z::AbstractVector{T};
        target_shock::Union{Int, Symbol} = 1,
        method::Symbol = :tsls) where {T <: AbstractFloat}
    return ExternalInstrument(reshape(Z, :, 1), target_shock; method = method)
end

# Positional backward compat: vector input
function ExternalInstrument(Z::AbstractVector{T}, target_shock::Int;
        method::Symbol = :tsls) where {T <: AbstractFloat}
    return ExternalInstrument(reshape(Z, :, 1), target_shock; method = method)
end

# --- ProxyIV constructors ----------------------------------------------------

"""
    ProxyIV(proxy::AbstractVector, target_shock; relevance_threshold=10.0)

Convenience constructor for a single proxy variable.
"""
function ProxyIV(proxy::AbstractVector{T}, target_shock::Int;
        relevance_threshold::Float64 = 10.0) where {T <: AbstractFloat}
    return ProxyIV(reshape(proxy, :, 1), [target_shock];
        relevance_threshold = relevance_threshold)
end

# --- Target resolution --------------------------------------------------------

"""
    _resolve_target(target, names) -> Int

Resolve a target shock specification to an integer index.
"""
function _resolve_target(target::Int, names::Vector{Symbol})
    1 <= target <= length(names) ||
        throw(ArgumentError("target_shock=$target out of range [1, $(length(names))]"))
    return target
end

function _resolve_target(target::Symbol, names::Vector{Symbol})
    idx = findfirst(==(target), names)
    idx === nothing &&
        throw(ArgumentError("target_shock=:$target not found. Available: $names"))
    return idx
end
