# ============================================================================
# IV-SVAR Diagnostics
# ============================================================================

"""
    first_stage_F(model::VARModel{T, <:IVSVAR}) where T

Return the first-stage F-statistic from a proxy-SVAR model.

A value above 10 is the conventional threshold for instrument relevance
(Stock, Wright & Yogo, 2002).
"""
function first_stage_F(model::VARModel{T, <:IVSVAR}) where {T}
    haskey(model.metadata, :first_stage_F) ||
        error("Model metadata does not contain first_stage_F")
    return model.metadata.first_stage_F
end

"""
    iv_summary(model::VARModel{T, <:IVSVAR}) where T

Print a summary of the proxy-SVAR first stage and impact coefficients.
"""
function iv_summary(model::VARModel{T, <:IVSVAR}) where {T}
    F = first_stage_F(model)
    target = model.metadata.target_shock
    β = model.metadata.iv_coefficients

    println("Proxy-SVAR Identification Summary")
    println("=" ^ 50)
    println("Target shock: variable $(model.names[target]) (index $target)")
    println("First-stage F-statistic: $(round(F; digits=2))",
        F < 10 ? "  ⚠ WEAK INSTRUMENT" : "  ✓")
    println("\nImpact coefficients (unit effect normalization):")
    for (i, name) in enumerate(model.names)
        println("  $name: $(round(β[i]; digits=4))")
    end
end

"""
    first_stage_F(model::VARModel, id::IVIdentification)

Compute the first-stage F-statistic from a reduced-form VAR and an instrument.

A value above 10 is the conventional threshold for instrument relevance
(Stock, Wright & Yogo, 2002).
"""
function first_stage_F(model::VARModel{T}, id::IVIdentification) where {T}
    ν = residuals(model)
    Z, target = _extract_instrument(id.instrument, size(ν, 1), n_lags(model), model.names)
    _, F_stat, _ = _iv_identify(ν, Z, target)
    return F_stat
end

"""
    iv_summary(model::VARModel, id::IVIdentification)

Print a summary of the SVAR-IV first stage and impact coefficients.
"""
function iv_summary(model::VARModel{T}, id::IVIdentification) where {T}
    ν = residuals(model)
    Z, target = _extract_instrument(id.instrument, size(ν, 1), n_lags(model), model.names)
    β_iv, F_stat, _ = _iv_identify(ν, Z, target)

    println("SVAR-IV Identification Summary")
    println("=" ^ 50)
    println("Target shock: variable $(model.names[target]) (index $target)")
    println("First-stage F-statistic: $(round(F_stat; digits=2))",
        F_stat < 10 ? "  ⚠ WEAK INSTRUMENT" : "  ✓")
    println("\nImpact coefficients (unit effect normalization):")
    for (i, name) in enumerate(model.names)
        println("  $name: $(round(β_iv[i]; digits=4))")
    end
end

export first_stage_F, iv_summary
