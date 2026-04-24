# ============================================================================
# VAR Simulation
# ============================================================================

"""
    simulate_var(model::VARModel, innovations, Y_init; burn_in=0)

Simulate VAR process given innovations and initial conditions.

# Arguments
- `model::VARModel`: Estimated VAR model (provides coefficients)
- `innovations::Matrix`: Innovation matrix (T × n_vars)
- `Y_init::Matrix`: Initial conditions (at least n_lags rows)

# Keyword Arguments
- `burn_in::Int=0`: Number of initial periods to discard

# Returns
- Simulated data matrix (T × n_vars)
"""
function simulate_var(model::VARModel{T}, innovations::AbstractMatrix{T},
        Y_init::AbstractMatrix{T}; burn_in::Int = 0) where {T}
    n_lags_val = n_lags(model)
    n_vars_val = n_vars(model)
    n_periods = size(innovations, 1)
    n_total = n_periods + burn_in

    Y_sim = zeros(T, n_total, n_vars_val)
    simulate_var!(Y_sim, model.coefficients.intercept, model.coefficients.lags,
        innovations, Y_init; burn_in = burn_in)

    # Return data after burn-in
    return burn_in == 0 ? Y_sim : Y_sim[(burn_in + 1):end, :]
end

"""
    simulate_var!(Y_sim, intercept, lags, innovations, Y_init; burn_in=0)

In-place VAR simulation. Writes into the caller-supplied `Y_sim` (size
`(n_periods + burn_in) × n_vars`) rather than allocating. Bit-for-bit
equivalent to the allocating `simulate_var`: same loop order, same `@inbounds`,
no BLAS reordering. The top `n_lags` rows of `Y_sim` are overwritten with the
last `n_lags` rows of `Y_init`, so any contents there on entry are ignored.

`Y_sim` must have exactly `(n_periods + burn_in)` rows, not the post-burn-in
`n_periods`; the caller is responsible for the size match. This lets the
bootstrap path reuse a single simulation buffer across every replication.
"""
function simulate_var!(
        Y_sim::AbstractMatrix{T},
        intercept::AbstractVector{T},
        lags::AbstractArray{T, 3},
        innovations::AbstractMatrix{T},
        Y_init::AbstractMatrix{T};
        burn_in::Int = 0) where {T}
    n_vars_val = size(intercept, 1)
    n_lags_val = size(lags, 3)
    n_periods = size(innovations, 1)
    n_total = n_periods + burn_in

    # Dimension checks
    size(lags, 1) == n_vars_val && size(lags, 2) == n_vars_val ||
        throw(ArgumentError("lags must be (n_vars, n_vars, n_lags)"))
    size(innovations, 2) == n_vars_val ||
        throw(ArgumentError("innovations must have $n_vars_val columns"))
    size(Y_init, 1) >= n_lags_val ||
        throw(ArgumentError("Y_init must have at least $n_lags_val rows"))
    size(Y_init, 2) == n_vars_val ||
        throw(ArgumentError("Y_init must have $n_vars_val columns"))
    size(Y_sim, 1) == n_total || throw(ArgumentError(
        "Y_sim must have $n_total rows (got $(size(Y_sim, 1)))"))
    size(Y_sim, 2) == n_vars_val || throw(ArgumentError(
        "Y_sim must have $n_vars_val columns (got $(size(Y_sim, 2)))"))

    # Seed the first n_lags rows from Y_init (preserving original scalar loop
    # order to match the allocating implementation bit-for-bit).
    @inbounds for j in 1:n_vars_val, i in 1:n_lags_val

        Y_sim[i, j] = Y_init[size(Y_init, 1) - n_lags_val + i, j]
    end

    # Simulate — loop structure and arithmetic identical to the allocating
    # implementation. Do not reorder.
    @inbounds for t in (n_lags_val + 1):n_total
        for j in 1:n_vars_val
            Y_sim[t, j] = intercept[j]
        end
        for lag in 1:n_lags_val
            for i in 1:n_vars_val        # from variable
                for j in 1:n_vars_val    # to variable
                    Y_sim[t, j] += lags[j, i, lag] * Y_sim[t - lag, i]
                end
            end
        end
        innov_idx = t - n_lags_val - burn_in
        if innov_idx > 0 && innov_idx <= n_periods
            for j in 1:n_vars_val
                Y_sim[t, j] += innovations[innov_idx, j]
            end
        end
    end
    return Y_sim
end

"""
    forecast(model::VARModel, h::Int; Y_init=nothing, include_draws=false, n_draws=1000)

Forecast h periods ahead from VAR model.

# Arguments
- `model::VARModel`: Estimated VAR model
- `h::Int`: Forecast horizon

# Keyword Arguments
- `Y_init::Matrix=nothing`: Initial conditions (default: use last n_lags observations)
- `include_draws::Bool=false`: Include forecast draws for uncertainty quantification
- `n_draws::Int=1000`: Number of forecast draws (if include_draws=true)

# Returns
- `forecast::Matrix`: Point forecast (h × n_vars)
- If `include_draws=true`: NamedTuple with `forecast`, `draws`, and `bands`
"""
function forecast(model::VARModel{T}, h::Int;
        Y_init::Union{Nothing, AbstractMatrix{T}} = nothing,
        include_draws::Bool = false,
        n_draws::Int = 1000,
        coverage::Vector{Float64} = [0.68, 0.90, 0.95]) where {T}
    h > 0 || throw(ArgumentError("forecast horizon h must be positive"))

    n_lags_val = n_lags(model)
    n_vars_val = n_vars(model)

    # Initial conditions
    if Y_init === nothing
        Y_init = model.Y[(end - n_lags_val + 1):end, :]
    end

    if !include_draws
        # Point forecast (conditional expectation)
        innovations_zero = zeros(T, h, n_vars_val)
        forecast_point = simulate_var(model, innovations_zero, Y_init)
        return forecast_point
    else
        # Forecast with uncertainty
        Σ = vcov(model)
        L = cholesky(Σ).L

        # Storage for draws
        forecast_draws = zeros(T, n_draws, h, n_vars_val)

        for d in 1:n_draws
            # Draw innovations from N(0, Σ)
            innovations = randn(T, h, n_vars_val) * L'
            forecast_draws[d, :, :] = simulate_var(model, innovations, Y_init)
        end

        # Point forecast (mean of draws)
        forecast_point = dropdims(mean(forecast_draws; dims = 1); dims = 1)

        # Confidence bands
        forecast_bands = compute_forecast_bands(forecast_draws, coverage)

        return (
            forecast = forecast_point,
            draws = forecast_draws,
            bands = forecast_bands,
            coverage = coverage
        )
    end
end

"""
    compute_forecast_bands(forecast_draws, coverage)

Compute forecast confidence bands from draws.
"""
function compute_forecast_bands(forecast_draws::Array{T, 3}, coverage::Vector{Float64}) where {T}
    n_coverage = length(coverage)
    h, n_vars = size(forecast_draws)[2:3]

    lower = zeros(T, n_coverage, h, n_vars)
    upper = zeros(T, n_coverage, h, n_vars)

    for (i, α) in enumerate(coverage)
        α_lower = (1 - α) / 2
        α_upper = 1 - α_lower

        for t in 1:h
            for j in 1:n_vars
                draws_tj = forecast_draws[:, t, j]
                lower[i, t, j] = quantile(draws_tj, α_lower)
                upper[i, t, j] = quantile(draws_tj, α_upper)
            end
        end
    end

    return (lower = lower, upper = upper)
end

# ============================================================================
# Simulation with Structural Shocks
# ============================================================================

"""
    simulate_structural(model::VARModel, identification::AbstractIdentification,
                       structural_shocks::Matrix, Y_init; burn_in=0)

Simulate VAR using structural shocks.

# Arguments
- `model::VARModel`: Estimated VAR model
- `identification::AbstractIdentification`: Identification scheme
- `structural_shocks::Matrix`: Structural shock matrix (T × n_shocks)
- `Y_init::Matrix`: Initial conditions

# Returns
- Simulated data matrix
"""
function simulate_structural(model::VARModel{T}, identification::AbstractIdentification,
        structural_shocks::AbstractMatrix{T},
        Y_init::AbstractMatrix{T}; burn_in::Int = 0) where {T}
    # Get impact matrix
    P = rotation_matrix(model, identification)

    # Convert structural to reduced-form shocks: u = P * ε
    innovations = structural_shocks * P'

    return simulate_var(model, innovations, Y_init; burn_in = burn_in)
end

# ============================================================================
# Historical Decomposition
# ============================================================================

"""
    historical_decomposition(model::VARModel, identification::AbstractIdentification)

Compute historical decomposition of observed data into structural shocks.

# Returns
- NamedTuple with `contributions` (T × n_vars × n_shocks) and `initial_condition` (T × n_vars)
"""
function historical_decomposition(model::VARModel{T}, identification::AbstractIdentification) where {T}
    n_lags_val = n_lags(model)
    n_vars_val = n_vars(model)
    n_obs_val = effective_obs(model)

    # Identify structural shocks
    P = rotation_matrix(model, identification)
    P_inv = inv(P)

    # Recover structural shocks from residuals
    residuals_mat = residuals(model)
    structural_shocks = residuals_mat * P_inv'  # (T-p) × n_vars

    # Preallocate contributions
    contributions = zeros(T, n_obs_val, n_vars_val, n_vars_val)

    # Compute MA representation
    Φ = compute_ma_matrices(model.companion, n_obs_val, n_vars_val, n_lags_val)

    # Historical decomposition: y_t = Σ_s Σ_h Φ_h * P * e_{s,t-h}
    for t in 1:n_obs_val
        for shock in 1:n_vars_val
            # Contribution of shock 'shock' to all variables at time t
            for h in 0:min(t - 1, size(Φ, 3) - 1)
                if t - h > 0
                    contrib = Φ[:, :, h + 1] * P[:, shock] * structural_shocks[t - h, shock]
                    contributions[t, :, shock] .+= contrib
                end
            end
        end
    end

    # Initial condition contribution (everything not explained by shocks)
    Y_actual = model.Y[(n_lags_val + 1):end, :]
    initial_condition = Y_actual - sum(contributions; dims = 3)[:, :, 1]

    return (
        contributions = contributions,
        initial_condition = initial_condition,
        structural_shocks = structural_shocks
    )
end

# ============================================================================
# Variance Decomposition
# ============================================================================

"""
    variance_decomposition(irf::IRFResult; horizon=nothing)

Compute forecast error variance decomposition from IRFs.

# Arguments
- `irf::IRFResult`: Computed impulse responses
- `horizon::Int=nothing`: Specific horizon (default: all horizons)

# Returns
- Array (horizon × n_vars × n_shocks) with variance shares (sum to 1 across shocks)
"""
function variance_decomposition(irf::IRFResult{T}; horizon_spec::Union{
        Nothing, Int} = nothing) where {T}
    irf_array = irf.irf  # (H+1, n_vars, n_shocks)
    H, n_vars_val, n_shocks = size(irf_array)

    # Compute MSE at each horizon
    mse = zeros(T, H, n_vars_val)
    for h in 1:H
        for j in 1:n_vars_val
            mse[h, j] = sum(irf_array[1:h, j, :] .^ 2)
        end
    end

    # Compute contribution of each shock
    variance_shares = zeros(T, H, n_vars_val, n_shocks)
    for h in 1:H
        for j in 1:n_vars_val
            for k in 1:n_shocks
                variance_shares[h, j, k] = sum(irf_array[1:h, j, k] .^ 2) / mse[h, j]
            end
        end
    end

    if horizon_spec !== nothing
        return variance_shares[horizon_spec, :, :]
    else
        return variance_shares
    end
end
