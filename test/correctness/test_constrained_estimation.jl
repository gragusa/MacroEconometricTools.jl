using Test
using MacroEconometricTools

@testset "Constrained OLS against equation-by-equation references" begin
    fixture = correctness_fixture()
    Y = fixture.sample
    reference = reference_ols(Y, 2)
    names = [:output, :price]

    zero = ZeroConstraint(:output, [:price], [1])
    model = fit(OLSVAR, Y, 2; names, constraints = [zero])
    free = [1, 2, 4, 5]
    expected_output = zeros(5)
    expected_output[free] = reference.X[:, free] \ Y[3:end, 1]
    @test [coef(model).intercept[1]; vec(coef(model).lags[1, :, :])] ≈
        expected_output atol = 1.0e-12
    @test coef(model).lags[1, 2, 1] == 0

    block = BlockExogeneity([:price], [:output])
    block_model = fit(OLSVAR, Y, 2; names, constraints = [block])
    block_free = [1, 2, 4]
    expected_block = zeros(5)
    expected_block[block_free] = reference.X[:, block_free] \ Y[3:end, 1]
    @test [coef(block_model).intercept[1]; vec(coef(block_model).lags[1, :, :])] ≈
        expected_block atol = 1.0e-12
    @test all(coef(block_model).lags[1, 2, :] .== 0)
end

@testset "Fixed-value constraints: restricted least squares" begin
    fixture = correctness_fixture()
    Y = fixture.sample
    reference = reference_ols(Y, 2)
    names = [:output, :price]
    X = reference.X
    y = Y[3:end, 1]

    # Fixed coefficient on price lag 1 (row 3 of B) in the output equation
    fixed = FixedConstraint(:output, :price, 1, 0.3)
    model = fit(OLSVAR, Y, 2; names, constraints = [fixed])
    free = [1, 2, 4, 5]
    expected = zeros(5)
    expected[3] = 0.3
    expected[free] = X[:, free] \ (y - X[:, 3] * 0.3)
    @test [coef(model).intercept[1]; vec(coef(model).lags[1, :, :])] ≈ expected atol = 1.0e-12
    # The price equation is unrestricted
    @test coef(model).lags[2, :, :] ≈ coef(fixture.model).lags[2, :, :] atol = 1.0e-12

    # Fixed and zero constraints together
    zero = ZeroConstraint(:output, [:price], [2])
    mixed = fit(OLSVAR, Y, 2; names, constraints = [fixed, zero])
    free = [1, 2, 4]
    expected = zeros(5)
    expected[3] = 0.3
    expected[free] = X[:, free] \ (y - X[:, 3] * 0.3)
    @test [coef(mixed).intercept[1]; vec(coef(mixed).lags[1, :, :])] ≈ expected atol = 1.0e-12
    @test coef(mixed).lags[1, 2, 2] == 0
end
