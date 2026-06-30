using Test
using ExplicitImports
using MacroEconometricTools

@testset "ExplicitImports" begin
    # `all_explicit_imports_are_public` determines public-ness via `Base.ispublic`
    # only on Julia 1.11+; on older Julia it falls back to `Base.isexported` and
    # false-positives on `public`-but-unexported bindings. Gate it accordingly.
    #
    # `all_qualified_accesses_are_public` is disabled: the Tables.jl interface
    # functions (`Tables.istable`, `Tables.rows`, `Tables.getcolumn`) are accessed
    # by qualification as Tables.jl intends, but Tables.jl declares none of them
    # `public`.
    test_explicit_imports(MacroEconometricTools;
        all_explicit_imports_are_public = VERSION >= v"1.11",
        all_qualified_accesses_are_public = false)
end
