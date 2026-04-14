module MacroEconometricToolsDataFramesExt

using DataFrames
using MacroEconometricTools

import MacroEconometricTools: VAR

"""
    VAR(df::DataFrame, variables::Vector{Symbol}, lags::Int; kwargs...)

Estimate a VAR model from a DataFrame by selecting the specified columns.

# Arguments
- `df::DataFrame`: Data source
- `variables::Vector{Symbol}`: Column names to include in the VAR
- `lags::Int`: Number of lags

# Keyword Arguments
All keyword arguments are forwarded to `fit(OLSVAR, ...)`.
See `fit(::Type{OLSVAR}, ...)` for details (e.g., `constraints`, `demean`).

# Returns
- `VARModel{Float64, OLSVAR}`

# Examples
```julia
using DataFrames, MacroEconometricTools

df = DataFrame(GDP = randn(200), Inflation = randn(200), Rate = randn(200))
var = VAR(df, [:GDP, :Inflation, :Rate], 4)

# With constraints
var = VAR(df, [:GDP, :Inflation], 2; demean = true)
```
"""
function VAR(df::DataFrame, variables::Vector{Symbol}, lags::Int; kwargs...)
    for v in variables
        hasproperty(df, v) ||
            throw(ArgumentError("Column :$v not found in DataFrame. " *
                                "Available: $(propertynames(df))"))
    end
    Y = Matrix{Float64}(df[:, variables])
    return fit(OLSVAR, Y, lags; names = variables, kwargs...)
end

end # module
