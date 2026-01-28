using Test

include("test_basic.jl")
include("test_delta_method.jl")
include("test_estimation_correctness.jl")
include("test_bootstrap_parallel.jl")
include("test_sign_restrictions.jl")
include("test_hub_types.jl")

# Aqua.jl quality assurance tests
include("Aqua.jl")
