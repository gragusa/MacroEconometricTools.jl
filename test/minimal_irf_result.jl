# Included by several test files; define the type and its methods only once.
if !isdefined(@__MODULE__, :MinimalIRFResult)
    using MacroEconometricTools

    # A subtype that implements only the required interface methods, with plain
    # arrays and integer names, to check that nothing depends on the concrete
    # hub types or on AxisArray axis names.
    struct MinimalIRFResult <: AbstractIRFResult{Float64}
        point::Array{Float64, 3}
        lower::Vector{Array{Float64, 3}}
        upper::Vector{Array{Float64, 3}}
    end
    MacroEconometricTools.point_estimate(r::MinimalIRFResult) = r.point
    MacroEconometricTools.lowerbounds(r::MinimalIRFResult) = r.lower
    MacroEconometricTools.upperbounds(r::MinimalIRFResult) = r.upper
    MacroEconometricTools.coverages(::MinimalIRFResult) = [0.68, 0.95]
    MacroEconometricTools.varnames(r::MinimalIRFResult) = collect(1:size(r.point, 1))
    MacroEconometricTools.shocknames(::MinimalIRFResult) = [:demand]
end
