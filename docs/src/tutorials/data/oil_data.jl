# Simulated Oil Market Data
# Based on the structure of Kilian (2009) oil market VAR
# Variables: Oil production growth, Real activity index, Real oil price (log)

"""
    generate_oil_market_data(; T=400, seed=123)

Generate simulated monthly oil market data similar to Kilian (2009).

Returns a named tuple with:
- data: Matrix of size T × 3 (oil_prod, real_activity, oil_price)
- dates: Vector of date strings
- names: Variable names

The DGP is a VAR(24) with parameters calibrated to roughly match
empirical oil market dynamics.
"""
function generate_oil_market_data(; T=400, seed=123)
    using Random, Dates
    Random.seed!(seed)

    # True parameters (calibrated to match oil market stylized facts)
    n_vars = 3
    n_lags = 24

    # Initialize
    Y = zeros(T + n_lags, n_vars)

    # AR coefficients (decay structure)
    A = zeros(n_vars, n_vars, n_lags)

    # Oil production: persistent, weak response to demand
    A[1,1,1] = 0.85
    A[1,1,2] = 0.10
    A[1,2,1] = 0.02  # Weak response to real activity
    A[1,3,6] = -0.05 # Delayed response to oil price

    # Real activity: moderate persistence
    A[2,2,1] = 0.60
    A[2,2,2] = 0.20
    A[2,1,3] = 0.05  # Oil production affects activity
    A[2,3,1:3] .= [-0.08, -0.05, -0.03]  # Oil price shocks slow activity

    # Real oil price: responds to supply and demand
    A[3,3,1] = 0.75
    A[3,3,2] = 0.15
    A[3,1,1:2] .= [-0.20, -0.10]  # Supply shocks
    A[3,2,1:2] .= [0.30, 0.15]    # Demand shocks

    # Structural impact matrix (Cholesky decomposition)
    # Ordering: production → activity → price
    P = [1.0  0.0  0.0;
         0.3  0.9  0.0;
         -0.5  0.8  1.2]

    # Generate structural shocks
    ε = randn(T + n_lags, n_vars)

    # Simulate VAR
    for t in (n_lags+1):(T+n_lags)
        # Autoregressive part
        for lag in 1:n_lags
            Y[t,:] += A[:,:,lag]' * Y[t-lag,:]
        end
        # Structural shock
        Y[t,:] += P * ε[t,:]
    end

    # Remove burn-in
    Y = Y[(n_lags+1):end, :]

    # Scale to realistic ranges
    # Oil production: % change, mean 0, std ~3%
    Y[:,1] = Y[:,1] .* 0.3

    # Real activity: index, mean 0, std ~20
    Y[:,2] = Y[:,2] .* 2.0

    # Oil price: log real price, mean 4 (~ $55/barrel), std ~0.4
    Y[:,3] = 4.0 .+ Y[:,3] .* 0.04

    # Generate dates (monthly, starting 1990-01)
    start_date = Date(1990, 1, 1)
    dates = [Dates.format(start_date + Month(i-1), "yyyy-mm") for i in 1:T]

    names = [:oil_prod_growth, :real_activity, :log_oil_price]

    return (data = Y, dates = dates, names = names,
            description = "Simulated oil market data (3 vars, $(T) obs)")
end

"""
    load_oil_data()

Convenience function to load the simulated oil market data.
"""
function load_oil_data()
    return generate_oil_market_data(T=456, seed=42)  # 38 years of monthly data
end
