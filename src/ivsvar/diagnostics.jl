# ============================================================================
# IV-SVAR Diagnostics
# ============================================================================

# Models from the deprecated `fit(IVSVAR, ...)` carry their instrument in the spec.
function first_stage_F(model::VARModel{T, <:IVSVAR}) where {T}
    first_stage_F(model, IVIdentification(model.spec.instrument))
end

function iv_summary(model::VARModel{T, <:IVSVAR}) where {T}
    iv_summary(model, IVIdentification(model.spec.instrument))
end

"""
    first_stage_F(model::VARModel, id::IVIdentification)

Compute the first-stage F-statistic from a reduced-form VAR and an instrument,
over the residual periods in which the instrument is observed.

A value above 10 is the conventional threshold for instrument relevance
(Stock, Wright & Yogo, 2002).
"""
function first_stage_F(model::VARModel{T}, id::IVIdentification) where {T}
    ν = residuals(model)
    Z, target = _extract_instrument(_resolve_iv(model, id).instrument, size(ν, 1), n_lags(model), model.names)
    _, F_stat, _ = _iv_identify(ν, Z, target)
    return F_stat
end

"""
    iv_summary(model::VARModel, id::IVIdentification)

Print a summary of the SVAR-IV first stage and impact coefficients.
"""
function iv_summary(model::VARModel{T}, id::IVIdentification) where {T}
    ν = residuals(model)
    Z, target = _extract_instrument(_resolve_iv(model, id).instrument, size(ν, 1), n_lags(model), model.names)
    β_iv, F_stat, _ = _iv_identify(ν, Z, target)

    println("SVAR-IV Identification Summary")
    println("=" ^ 50)
    println("Target shock: variable $(model.names[target]) (index $target)")
    println("Instrument observed in $(length(_observed_rows(Z))) of $(size(ν, 1)) residual periods")
    println("First-stage F-statistic: $(round(F_stat; digits=2))",
        F_stat < 10 ? "  ⚠ WEAK INSTRUMENT" : "  ✓")
    println("\nImpact coefficients (unit effect normalization):")
    for (i, name) in enumerate(model.names)
        println("  $name: $(round(β_iv[i]; digits=4))")
    end
end

export first_stage_F, iv_summary
