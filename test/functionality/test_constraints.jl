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
        AbstractConstraint[ZeroConstraint(:y, [:x], [1])], names, 2
    )
    @test size(selection) == (10, 9)
    @test n_free == 9
    @test sum(selection; dims = 1) == ones(1, 9)

    @test MacroEconometricTools.check_constraints(constraints, names, 2)
    @test_throws ArgumentError MacroEconometricTools.check_constraints(
        AbstractConstraint[ZeroConstraint(:missing, [:x])], names, 2
    )
    @test_throws ArgumentError MacroEconometricTools.check_constraints(
        AbstractConstraint[FixedConstraint(:y, :x, 3, 1.0)], names, 2
    )
    @test_throws ArgumentError MacroEconometricTools.apply_constraints!(
        copy(matrix),
        AbstractConstraint[ZeroConstraint(:y, [:missing])], names, 2
    )
end

struct UnsupportedConstraint <: AbstractConstraint end

@testset "Constraint resolution errors" begin
    names = [:y, :x]
    check(c) = MacroEconometricTools.check_constraints(AbstractConstraint[c...], names, 2)
    @test_throws "constraint variable :z is not a model variable" check([ZeroConstraint(:z, [:x])])
    @test_throws "constraint regressor :z is not a model variable" check([ZeroConstraint(:y, [:z])])
    @test_throws "constraint lag 3 is outside 1:2" check([ZeroConstraint(:y, [:x], [3])])
    @test_throws "constraint lag 0 is outside 1:2" check([FixedConstraint(:y, :x, 0, 1.0)])
    @test_throws "constraint 'from' variable :z" check([BlockExogeneity([:z], [:y])])
    @test_throws "constraint 'to' variable :z" check([BlockExogeneity([:x], [:z])])
    @test_throws "conflicting constraints" check(
        [ZeroConstraint(:y, [:x], [1]), FixedConstraint(:y, :x, 1, 0.5)]
    )
    @test_throws "conflicting constraints" check(
        [FixedConstraint(:y, :x, 1, 0.5), FixedConstraint(:y, :x, 1, 0.4)]
    )
    # Overlapping constraints with the same value are consistent
    @test check([ZeroConstraint(:y, [:x], [1]), BlockExogeneity([:x], [:y])])

    @test_throws "unsupported constraint type" check([UnsupportedConstraint()])

    # Every path resolves through the same rules, including estimation
    Y = randn(50, 2)
    @test_throws "constraint lag 3 is outside 1:2" MacroEconometricTools.constrained_ols(
        MacroEconometricTools.create_lags(Y, 2)[3:end, :], Y[3:end, :],
        AbstractConstraint[ZeroConstraint(:y, [:x], [3])], names, 2, 2
    )
    @test_throws "constraint regressor :z" MacroEconometricTools.build_selection_matrix(
        AbstractConstraint[ZeroConstraint(:y, [:z])], names, 2
    )

    # Fixed coefficients are not free parameters
    _, n_free = MacroEconometricTools.build_selection_matrix(
        AbstractConstraint[FixedConstraint(:y, :x, 1, 0.5)], names, 2
    )
    @test n_free == 9
end
