include("../../../lib/gaussian.jl")
include("../../../lib/distributionbag.jl")
include("../../../lib/utils.jl")
include("../../../lib/factors.jl")
include("generate_data_gaussian_mean_factor.jl")
# include("../train_nn.jl")

using Distributions
using StatsBase
using .GaussianMeanFactorGeneration: generate_dataset_gaussian_mean_factor, X, Y, dimension, mean, variance, remove_variable, get_variable
using .GaussianDistribution: Gaussian1D, Gaussian1DFromMeanVariance, mean, variance, absdiff, logNormProduct, logNormRatio
  
function update_marginal_x!(input)
    msgBackX, msgBackY, beta_squared = Gaussian1D(input[1], input[2]), Gaussian1D(input[3], input[4]), input[5]^2
    c = 1 / (1 + beta_squared * msgBackY.rho)
    newMsg = Gaussian1D(msgBackY.tau * c, msgBackY.rho * c)

    return msgBackX * newMsg
end
  
function update_marginal_y!(input)
    msgBackX, msgBackY, beta_squared = Gaussian1D(input[1], input[2]), Gaussian1D(input[3], input[4]), input[5]^2
    c = 1 / (1 + beta_squared * msgBackX.rho)
    newMsg = Gaussian1D(msgBackX.tau * c, msgBackX.rho * c)
  
    return msgBackY * newMsg
end

function analytical_update_uniform_y(input)
    updated_marginal_x = update_marginal_x!(input)
    updated_marginal_y = update_marginal_y!(input)
    return nat_to_ls([updated_marginal_x.tau, updated_marginal_x.rho, updated_marginal_y.tau, updated_marginal_y.rho, input[5]])
end

function ls_to_nat(input)
    return [input[1] / input[2], 1.0 / input[2], input[3] / input[4], 1.0 / input[4], input[5]]
end

nat_to_ls(input) = ls_to_nat(input)


function naive_sample_update_uniform_y(input, n=1_000_000)
    ls_input = nat_to_ls(input)
    samples_x = rand(Normal(ls_input[1], sqrt(ls_input[2])), n)
    samples_y = deepcopy(samples_x)
    return [StatsBase.mean(samples_x), StatsBase.var(samples_x), StatsBase.mean(samples_y), StatsBase.var(samples_y), ls_input[5]]
end

function sample_update_uniform_y(input, n=1_000_000)
    ls_input = nat_to_ls(input)
    samples_x = rand(Normal(ls_input[1], sqrt(ls_input[2])), n)
    samples_y = [rand(Normal(s, ls_input[5])) for s in samples_x]
    weights = Weights([pdf(Normal(samples_x[i], ls_input[5]), samples_y[i]) for i in 1:n])
    mean_y = StatsBase.mean(samples_y)
    var_y = StatsBase.var(samples_y; mean=mean_y)
    return [StatsBase.mean(samples_x), StatsBase.var(samples_x), mean_y, var_y, ls_input[5]]
end

input_nat = [0.2, 0.2, 0.0, 0.0, 2.0]
num_samples = 100_000_000


println("Input LS: ", [1.0, 5.0, 0.0, Inf, 2.0])
println("Analytical update: ", analytical_update_uniform_y(input_nat))
println("Sampled update: ", sample_update_uniform_y(input_nat, num_samples))
println("Naive sampled update: ", naive_sample_update_uniform_y(input_nat, num_samples))

function half_variance(m, v)
    samples = rand(Normal(m, sqrt(v)), 100_000_000)
    weights = Weights([pdf(Normal(m, sqrt(v)), s) for s in samples])
    return StatsBase.mean(samples, weights), StatsBase.var(samples, weights; mean=StatsBase.mean(samples, weights))
end