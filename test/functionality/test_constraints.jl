using Test
using MacroEconometricTools

@testset "Constraint utilities and validation" begin
    names = [:y, :x]
    matrix = reshape(collect(1.0:10.0), 2, 5)
    constraints = AbstractConstraint[
        ZeroConstraint(:y, [:x]),
        FixedConstraint(:x, :x, 2, 0.25),
        BlockExogeneity([:y], [:x])
    ]
    MacroEconometricTools.apply_constraints!(matrix, constraints, names, 2)
    @test matrix[1, [3, 5]] == [0, 0]
    @test matrix[2, 5] == 0.25
    @test matrix[2, [2, 4]] == [0, 0]

    selection, n_free = MacroEconometricTools.build_selection_matrix(
        AbstractConstraint[ZeroConstraint(:y, [:x], [1])], names, 2)
    @test size(selection) == (10, 9)
    @test n_free == 9
    @test sum(selection; dims = 1) == ones(1, 9)

    @test MacroEconometricTools.check_constraints(constraints, names, 2)
    @test_throws ArgumentError MacroEconometricTools.check_constraints(
        AbstractConstraint[ZeroConstraint(:missing, [:x])], names, 2)
    @test_throws ArgumentError MacroEconometricTools.check_constraints(
        AbstractConstraint[FixedConstraint(:y, :x, 3, 1.0)], names, 2)
    @test_throws ArgumentError MacroEconometricTools.apply_constraints!(copy(matrix),
        AbstractConstraint[ZeroConstraint(:y, [:missing])], names, 2)
end
