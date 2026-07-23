using LinearAlgebra
using MacroEconometricTools
using StableRNGs: StableRNG

"""Return one deterministic, stable VAR(2) sample and its fitted model."""
function correctness_fixture(; n_obs = 600, burn_in = 100)
    rng = StableRNG(0x5eed)
    intercept = [0.15, -0.08]
    lags = cat([0.42 0.12; -0.08 0.31], [0.11 -0.05; 0.04 0.09]; dims = 3)
    impact = [0.8 0.0; 0.25 0.6]
    Y = zeros(n_obs + burn_in, 2)
    for t in 3:size(Y, 1)
        Y[t, :] = intercept + lags[:, :, 1] * Y[t - 1, :] +
                  lags[:, :, 2] * Y[t - 2, :] + impact * randn(rng, 2)
    end
    sample = Y[(burn_in + 1):end, :]
    return (; sample, model = fit(OLSVAR, sample, 2; names = [:output, :price]))
end

"""Independent OLS calculation which does not call the package estimator."""
function reference_ols(Y, p)
    n_obs, n = size(Y)
    X = ones(n_obs - p, 1 + n * p)
    for lag in 1:p
        X[:, (2 + (lag - 1) * n):(1 + lag * n)] = Y[(p + 1 - lag):(end - lag), :]
    end
    target = Y[(p + 1):end, :]
    coefficients = X \ target
    innovations = target - X * coefficients
    covariance = innovations' * innovations / (size(X, 1) - size(X, 2))
    return (; X, coefficients, innovations, covariance)
end

function unvech(lower::AbstractVector, n::Int)
    matrix = zeros(eltype(lower), n, n)
    k = 1
    for column in 1:n, row in column:n

        matrix[row, column] = lower[k]
        matrix[column, row] = lower[k]
        k += 1
    end
    return matrix
end

"""Compute Cholesky IRFs independently of package IRF and Jacobian code."""
function reference_irf(alpha, sigma_half, n, p, horizon)
    companion = zeros(eltype(alpha), n * p, n * p)
    companion[1:n, :] = reshape(alpha, n, n * p)
    if p > 1
        companion[(n + 1):end, 1:(n * (p - 1))] = Matrix{eltype(alpha)}(I, n * (p - 1), n *
                                                                                        (p -
                                                                                         1))
    end
    impact = Matrix(cholesky(Symmetric(unvech(sigma_half, n))).L)
    selection = [Matrix{eltype(alpha)}(I, n, n) zeros(eltype(alpha), n, n * (p - 1))]
    responses = zeros(eltype(alpha), horizon + 1, n, n)
    power = Matrix{eltype(alpha)}(I, n * p, n * p)
    for h in 0:horizon
        responses[h + 1, :, :] = selection * power * selection' * impact
        power *= companion
    end
    return responses
end

function central_jacobian(f, x; relative_step = 1e-5)
    jacobian = zeros(eltype(x), length(vec(f(x))), length(x))
    for j in eachindex(x)
        step = relative_step * max(abs(x[j]), one(eltype(x)))
        above, below = copy(x), copy(x)
        above[j] += step
        below[j] -= step
        jacobian[:, j] = (vec(f(above)) - vec(f(below))) / (2step)
    end
    return jacobian
end
