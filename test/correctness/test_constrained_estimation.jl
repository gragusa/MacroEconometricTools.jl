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
          expected_output atol = 1e-12
    @test coef(model).lags[1, 2, 1] == 0

    block = BlockExogeneity([:price], [:output])
    block_model = fit(OLSVAR, Y, 2; names, constraints = [block])
    block_free = [1, 2, 4]
    expected_block = zeros(5)
    expected_block[block_free] = reference.X[:, block_free] \ Y[3:end, 1]
    @test [coef(block_model).intercept[1]; vec(coef(block_model).lags[1, :, :])] ≈
          expected_block atol = 1e-12
    @test all(coef(block_model).lags[1, 2, :] .== 0)
end
