# ============================================================================
# IV-SVAR Instrument Utilities
# ============================================================================
# The type is defined in types.jl (ExternalInstrument)

# --- ExternalInstrument constructors ------------------------------------------

_target_shock(target::Integer) = Int(target)
_target_shock(target::Symbol) = target

function ExternalInstrument(Z::AbstractMatrix, target_shock::Union{Integer, Symbol})
    target = _target_shock(target_shock)
    S = nonmissingtype(eltype(Z))
    S === Union{} && throw(ArgumentError("instrument has no observed values"))
    return ExternalInstrument{float(S), typeof(target)}(Z, target)
end

function ExternalInstrument(Z::AbstractVector, target_shock::Union{Integer, Symbol})
    return ExternalInstrument(reshape(Z, :, 1), target_shock)
end

function ExternalInstrument(Z::AbstractVecOrMat; target_shock::Union{Integer, Symbol} = 1)
    return ExternalInstrument(Z, target_shock)
end

Base.@deprecate_binding ProxyIV ExternalInstrument false

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
