include("projectile_estimation_factorgraph.jl")
include("projectile_estimation_factorgraph_approximated.jl")
using .ProjectileEstimation
using DataStructures

true_data = OrderedDict(0.0 => 1.0, 1.0 =>6.0, 2.0 => 11.0, 3.0 => 16.0, 4.0 => 21.0, 5.0 => 26.0, 6.0 => 31.0, 7.0 => 36.0)
noisy_data = add_noise(true_data)
println("true data: ", true_data)
println("noisy data: ", noisy_data)


# function print_estimation_results_x(data, true_data)
#     estimate = ProjectileEstimation.estimate_initial_state_x!(data=data)
#     # print out the type of the estimate variable
#     println("type of estimate: ", typeof(estimate))

#     timestamps_used = [datapoint.first for datapoint in data]
#     println("timestamps used: ", timestamps_used)
#     println("final variables: ", estimate)
#     println("estimation_loss: ", ProjectileEstimation.estimation_loss(true_data, estimate))
#     println("----------------")
# end


# shortened_true_data = OrderedDict(collect(true_data)[1:4])
# print_estimation_results_x(OrderedDict(collect(true_data)[1:4]), shortened_true_data)
# print_estimation_results_x(OrderedDict(collect(noisy_data)[1:4]), shortened_true_data)

# print_estimation_results_x(OrderedDict(collect(true_data)[1:end]), true_data)
# print_estimation_results_x(OrderedDict(collect(noisy_data)[1:end]), true_data)

# print_estimation_results_x(OrderedDict(collect(true_data)[4:end]), true_data)
# print_estimation_results_x(OrderedDict(collect(noisy_data)[4:end]), true_data)

# println("################################")
# true_data = OrderedDict(0.0 => 1.0, 1.0 => 36.1, 2.0 => 61.4, 3.0 => 76.9, 4.0 => 82.6, 5.0 => 78.5, 6.0 => 64.6, 7.0 => 40.9)
# noisy_data = add_noise(true_data)
# println("true data: ", true_data)
# println("noisy data: ", noisy_data)

# function print_estimation_results_y(data, true_data)
#     estimate =ProjectileEstimation.estimate_initial_state_x!(data=data, factor_coordinate=(t -> t), bias_coordinate=(t -> 0.5*(-9.8)*t^2), factor_velocity=(t -> 1.0), bias_velocity=(t -> t * (-9.8)))
#     timestamps_used = [datapoint.first for datapoint in data]
#     println("timestamps used: ", timestamps_used)
#     println("final variables: ", estimate)
#     println("estimation_loss: ", ProjectileEstimation.estimation_loss(true_data, estimate))
#     println("----------------")
# end


# shortened_true_data = OrderedDict(collect(true_data)[1:4])
# print_estimation_results_y(OrderedDict(collect(true_data)[1:4]), shortened_true_data)
# print_estimation_results_y(OrderedDict(collect(noisy_data)[1:4]), shortened_true_data)

# print_estimation_results_y(OrderedDict(collect(true_data)[1:end]), true_data)
# print_estimation_results_y(OrderedDict(collect(noisy_data)[1:end]), true_data)

# print_estimation_results_y(OrderedDict(collect(true_data)[4:end]), true_data)
# print_estimation_results_y(OrderedDict(collect(noisy_data)[4:end]), true_data)


println("##########################################################################")
println("##########################################################################")
println("##########################################################################")


function print_estimation_results_x_approximate(data, true_data)
    estimate = ProjectileEstimationApproximated.estimate_initial_state_x!(data=data)
    # print out the type of the estimate variable
    println("type of estimate: ", typeof(estimate))

    timestamps_used = [datapoint.first for datapoint in data]
    println("timestamps used: ", timestamps_used)
    println("final variables: ", estimate)
    println("estimation_loss: ", ProjectileEstimationApproximated.estimation_loss(true_data, estimate))
    println("----------------")
end


shortened_true_data = OrderedDict(collect(true_data)[1:4])
print_estimation_results_x_approximate(OrderedDict(collect(true_data)[1:4]), shortened_true_data)
print_estimation_results_x_approximate(OrderedDict(collect(noisy_data)[1:4]), shortened_true_data)

print_estimation_results_x_approximate(OrderedDict(collect(true_data)[1:end]), true_data)
print_estimation_results_x_approximate(OrderedDict(collect(noisy_data)[1:end]), true_data)

print_estimation_results_x_approximate(OrderedDict(collect(true_data)[4:end]), true_data)
print_estimation_results_x_approximate(OrderedDict(collect(noisy_data)[4:end]), true_data)

println("################################")
true_data = OrderedDict(0.0 => 1.0, 1.0 => 36.1, 2.0 => 61.4, 3.0 => 76.9, 4.0 => 82.6, 5.0 => 78.5, 6.0 => 64.6, 7.0 => 40.9)
noisy_data = add_noise(true_data)
println("true data: ", true_data)
println("noisy data: ", noisy_data)

function print_estimation_results_y_approximate(data, true_data)
    estimate =ProjectileEstimationApproximated.estimate_initial_state_x!(data=data, factor_coordinate=(t -> t), bias_coordinate=(t -> 0.5*(-9.8)*t^2), factor_velocity=(t -> 1.0), bias_velocity=(t -> t * (-9.8)))
    timestamps_used = [datapoint.first for datapoint in data]
    println("timestamps used: ", timestamps_used)
    println("final variables: ", estimate)
    println("estimation_loss: ", ProjectileEstimationApproximated.estimation_loss(true_data, estimate))
    println("----------------")
end


shortened_true_data = OrderedDict(collect(true_data)[1:4])
print_estimation_results_y_approximate(OrderedDict(collect(true_data)[1:4]), shortened_true_data)
print_estimation_results_y_approximate(OrderedDict(collect(noisy_data)[1:4]), shortened_true_data)

print_estimation_results_y_approximate(OrderedDict(collect(true_data)[1:end]), true_data)
print_estimation_results_y_approximate(OrderedDict(collect(noisy_data)[1:end]), true_data)

print_estimation_results_y_approximate(OrderedDict(collect(true_data)[4:end]), true_data)
print_estimation_results_y_approximate(OrderedDict(collect(noisy_data)[4:end]), true_data)