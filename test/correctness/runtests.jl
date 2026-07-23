# Tests in this layer answer: "Are the numerical results right?"
# Prefer independent formulas, published/reference values, and cross-language fixtures.
include("helpers.jl")
include("test_ols_irf_oracles.jl")
include("test_constrained_estimation.jl")
include("../test_estimation_correctness.jl")
include("../test_delta_method.jl")
include("../test_proxy_svar_crossval.jl")
include("../test_proxy_svar_bootstrap.jl")
