using Test
using Aqua
using MacroEconometricTools

@testset "Aqua.jl" begin
    Aqua.test_all(MacroEconometricTools)
end
