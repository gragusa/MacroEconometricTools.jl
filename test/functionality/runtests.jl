# Tests in this layer answer: "Does the public surface behave as promised?"
# This includes API contracts, types, errors, reproducibility, and integration paths.
include("../test_basic.jl")
include("test_constraints.jl")
include("test_plot_recipes.jl")
include("../test_bootstrap_parallel.jl")
include("../test_sign_restrictions.jl")
include("../test_hub_types.jl")
include("../test_proxy_svar.jl")
include("../test_ivsvar_new_api.jl")
