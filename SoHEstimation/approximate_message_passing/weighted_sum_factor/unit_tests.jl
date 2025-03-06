include("../../../lib/gaussian.jl")
include("generate_data_weighted_sum_factor.jl")
include("plot_nn_results.jl")

using Test
using StatsBase
using Distributions

using Test

# @testset "Variable enum functions tests" begin
#     @testset "dimension function" begin
#         @test dimension(X) == 1
#         @test dimension(Y) == 3
#         @test dimension(Z) == 5
#         @test dimension(X, 2) == 2
#         @test dimension(Y, 2) == 4
#         @test dimension(Z, 2) == 6
#         @test_throws AssertionError dimension(X, 3)
#     end

#     @testset "mean function" begin
#         x9 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#         x6 = [1, 2, 3, 4, 5, 6]
#         @test mean(X, x9) == 1
#         @test mean(Y, x9) == 3
#         @test mean(Z, x9) == 5
#         @test mean(X, x6) == 1
#         @test mean(Y, x6) == 3
#         @test mean(Z, x6) == 5
#         @test_throws AssertionError mean(X, [1, 2, 3])
#     end

#     @testset "variance function" begin
#         x9 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#         x6 = [1, 2, 3, 4, 5, 6]
#         @test variance(X, x9) == 2
#         @test variance(Y, x9) == 4
#         @test variance(Z, x9) == 6
#         @test variance(X, x6) == 2
#         @test variance(Y, x6) == 4
#         @test variance(Z, x6) == 6
#         @test_throws AssertionError variance(X, [1, 2, 3])
#     end

#     @testset "remove_variable function" begin
#         x9 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#         x6 = [1, 2, 3, 4, 5, 6]
#         @test remove_variable(X, x9) == [3, 4, 5, 6, 7, 8, 9]
#         @test remove_variable(Y, x9) == [1, 2, 5, 6, 7, 8, 9]
#         @test remove_variable(Z, x9) == [1, 2, 3, 4, 7, 8, 9]
#         @test remove_variable(X, x6) == [3, 4, 5, 6]
#         @test remove_variable(Y, x6) == [1, 2, 5, 6]
#         @test remove_variable(Z, x6) == [1, 2, 3, 4]
#         @test_throws AssertionError remove_variable(X, [1, 2, 3])
#     end

#     @testset "get_variable function" begin
#         x9 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#         x6 = [1, 2, 3, 4, 5, 6]
#         @test get_variable(X, x9) == [1, 2]
#         @test get_variable(Y, x9) == [3, 4]
#         @test get_variable(Z, x9) == [5, 6]
#         @test get_variable(X, x6) == [1, 2]
#         @test get_variable(Y, x6) == [3, 4]
#         @test get_variable(Z, x6) == [5, 6]
#         @test_throws AssertionError get_variable(X, [1, 2, 3])
#     end

#     @testset "a, b, c functions" begin
#         x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#         @test a(x) == 7
#         @test b(x) == 8
#         @test c(x) == 9
#         @test_throws AssertionError a([1, 2, 3, 4, 5, 6])
#         @test_throws AssertionError b([1, 2, 3, 4, 5, 6])
#         @test_throws AssertionError c([1, 2, 3, 4, 5, 6])
#     end


#     @testset "comparison of remove_variable and get_variable to script" begin
#         function remvar(variable::Variable, input, output)
#             if variable == X
#                 samples = input[3:end]
#                 targets = output[1:2]
#             elseif variable == Y
#                 samples = vcat(input[1:2], input[5:end])
#                 targets = output[3:4]
#             elseif variable == Z
#                 samples = vcat(input[1:4], input[7:end])
#                 targets = output[5:6]
#             end
#             return sample, target
#         end

#         for v in [X, Y, Z]
#             sample = [i for i in 1:9]
#             target = [i for i in 1:6]

#             @test (remove_variable(v, sample), get_variable(v, target)) == remvar(v, sample, target)
#         end
#     end
# end

@testset "Log-likelihood Weighted Statistics Tests" begin
    @testset "Log-likelihood Weighted Mean" begin
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        w = [-1.0, -2.0, -3.0, -4.0, -5.0]  # Log-likelihood weights
        weights = Weights(exp.(w))  # Convert to Weights object for StatsBase functions

        # Test with positive values
        @test isapprox(log_weighted_mean(x, w), StatsBase.mean(x, weights), atol=1e-6)

        # Test with negative values
        x_neg = [-5.0, -4.0, -3.0, -2.0, -1.0]
        @test isapprox(log_weighted_mean(x_neg, w), StatsBase.mean(x_neg, weights), atol=1e-6)

        # Test with mixed positive and negative values
        x_mixed = [-2.0, -1.0, 0.0, 1.0, 2.0]
        @test isapprox(log_weighted_mean(x_mixed, w), StatsBase.mean(x_mixed, weights), atol=1e-6)

        # Test with mixed positive and negative weights
        w_mixed = [-2.0, -1.0, 0.0, 1.0, 2.0]
        weights_mixed = Weights(exp.(w_mixed))
        @test isapprox(log_weighted_mean(x, w_mixed), StatsBase.mean(x, weights_mixed), atol=1e-6)

        # Test with mixed positive and negative weights and values
        @test isapprox(log_weighted_mean(x_mixed, w_mixed), StatsBase.mean(x_mixed, weights_mixed), atol=1e-6)
    end

    @testset "Log-likelihood Weighted Variance" begin
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        w = [-1.0, -2.0, -3.0, -4.0, -5.0]  # Log-likelihood weights
        weights = Weights(exp.(w))  # Convert to Weights object for StatsBase functions

        # Test with positive values
        @test isapprox(log_weighted_var(x, w), StatsBase.var(x, weights; corrected=false), atol=1e-6)

        # Test with negative values
        x_neg = [-5.0, -4.0, -3.0, -2.0, -1.0]
        @test isapprox(log_weighted_var(x_neg, w), StatsBase.var(x_neg, weights; corrected=false), atol=1e-6)

        # Test with mixed positive and negative values
        x_mixed = [-2.0, -1.0, 0.0, 1.0, 2.0]
        @test isapprox(log_weighted_var(x_mixed, w), StatsBase.var(x_mixed, weights; corrected=false), atol=1e-6)

        # Test with mixed positive and negative weights
        w_mixed = [-2.0, -1.0, 0.0, 1.0, 2.0]
        weights_mixed = Weights(exp.(w_mixed))
        @test isapprox(log_weighted_var(x, w_mixed), StatsBase.var(x, weights_mixed), atol=1e-6)

        # Test with mixed positive and negative weights and values
        w_mixed = [-2.0, -1.0, 0.0, 1.0, 2.0]
        weights_mixed = Weights(exp.(w_mixed))
        @test isapprox(log_weighted_var(x_mixed, w_mixed), StatsBase.var(x_mixed, weights_mixed), atol=1e-6)

        # Test with provided mean
        mean_value = log_weighted_mean(x, w)
        @test isapprox(log_weighted_var(x, w; mean=mean_value),
            StatsBase.var(x, weights; corrected=false), atol=1e-6)
    end
end

@testset "sample_uniform_update tests" begin
    # Test when X is uniform
    @testset "X is uniform" begin
        input = to_tau_rho([0.0, 0.0, 2.0, 1.0, 3.0, 1.0, 1.0, 2.0, 3.0])
        set_tau!(X, input, 0.0)
        set_rho!(X, input, 0.0)
        result = sample_uniform_update(input, 10000)
        
        @test length(result) == 6
        @test result[1] != 0.0  # tau_x should be updated
        @test result[2] != 0.0  # rho_x should be updated
        @test result[3] == 2.0  # tau_y should remain the same
        @test result[4] == 1.0  # rho_y should remain the same
        @test result[5] == 3.0  # tau_z should remain the same
        @test result[6] == 1.0  # rho_z should remain the same
        
        # Check if the sampled mean and variance of X are close to the expected values
        expected_mean_x = (3.0 - 2.0 * 2.0 - 3.0) / 1.0  # (Z - Y*b - c) / a
        expected_var_x = 1.0 / 1.0 + 1.0 * 4.0  # var(Z)/a^2 + var(Y)*(b/a)^2
        @test isapprox(result[1] / result[2], expected_mean_x, rtol=0.05)
        @test isapprox(1 / result[2], expected_var_x, rtol=0.05)
    end

    # Test when Y is uniform
    @testset "Y is uniform" begin
        input = to_tau_rho([1.0, 1.0, 0.0, 0.0, 3.0, 1.0, 1.0, 2.0, 3.0])
        set_tau!(Y, input, 0.0)
        set_rho!(Y, input, 0.0)
        result = sample_uniform_update(input, 10000)
        
        @test length(result) == 6
        @test result[1] == 1.0  # tau_x should remain the same
        @test result[2] == 1.0  # rho_x should remain the same
        @test result[3] != 0.0  # tau_y should be updated
        @test result[4] != 0.0  # rho_y should be updated
        @test result[5] == 3.0  # tau_z should remain the same
        @test result[6] == 1.0  # rho_z should remain the same
        
        # Check if the sampled mean and variance of Y are close to the expected values
        expected_mean_y = (3.0 - 1.0 * 1.0 - 3.0) / 2.0  # (Z - X*a - c) / b
        expected_var_y = 1.0 / 4.0 + 1.0 / 4.0  # var(Z)/b^2 + var(X)*(a/b)^2
        @test isapprox(result[3] / result[4], expected_mean_y, rtol=0.05)
        @test isapprox(1 / result[4], expected_var_y, rtol=0.05)
    end

    # Test when Z is uniform
    @testset "Z is uniform" begin
        input = to_tau_rho([1.0, 1.0, 2.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0])
        set_tau!(Z, input, 0.0)
        set_rho!(Z, input, 0.0)
        result = sample_uniform_update(input, 10000)
        
        @test length(result) == 6
        @test result[1] == 1.0  # tau_x should remain the same
        @test result[2] == 1.0  # rho_x should remain the same
        @test result[3] == 2.0  # tau_y should remain the same
        @test result[4] == 1.0  # rho_y should remain the same
        @test result[5] != 0.0  # tau_z should be updated
        @test result[6] != 0.0  # rho_z should be updated
        
        # Check if the sampled mean and variance of Z are close to the expected values
        expected_mean_z = 1.0 * 1.0 + 2.0 * 2.0 + 3.0  # X*a + Y*b + c
        expected_var_z = 1.0 + 4.0  # var(X)*a^2 + var(Y)*b^2
        @test isapprox(result[5] / result[6], expected_mean_z, rtol=0.05)
        @test isapprox(1 / result[6], expected_var_z, rtol=0.05)
    end

    # Test error handling
    @testset "Error handling" begin
        @test_throws AssertionError sample_uniform_update([1.0, 1.0, 2.0, 1.0, 3.0], 1000)  # Invalid input length
        @test_throws AssertionError sample_uniform_update([1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 1.0, 2.0, 3.0], 0)  # Invalid samples_per_input
    end
end