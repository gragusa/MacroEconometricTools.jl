using Test

@testset "Correctness" begin
    include("correctness/runtests.jl")
end

@testset "Functionality" begin
    include("functionality/runtests.jl")
end

# Aqua.jl quality assurance tests
include("Aqua.jl")

# ExplicitImports.jl import-hygiene checks
include("ExplicitImports.jl")
