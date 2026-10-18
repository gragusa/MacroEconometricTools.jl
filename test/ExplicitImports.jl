using Test
using ExplicitImports
using MacroEconometricTools

@testset "ExplicitImports" begin
    # `all_qualified_accesses_are_public` is disabled: the Tables.jl interface
    # functions (`Tables.istable`, `Tables.rows`, `Tables.getcolumn`) are accessed
    # by qualification as Tables.jl intends, but Tables.jl declares none of them
    # `public`.
    test_explicit_imports(
        MacroEconometricTools;
        all_qualified_accesses_are_public = false
    )
end
