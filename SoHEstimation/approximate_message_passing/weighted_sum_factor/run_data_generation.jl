include("../../../lib/gaussian.jl")
include("../../../lib/distributionbag.jl")
include("../../../lib/factors.jl")
include("../../../lib/utils.jl")
include("generate_data_weighted_sum_factor.jl")

using Distributions
using .WeightedSumFactorGeneration: to_tau_rho, to_mean_variance, tau, rho, get_variable, dimension, generate_output_weighted_sum_factor

dataset = WeightedSumFactorGeneration.generate_dataset_weighted_sum_factor(
    n=                          500,    
    patience=                   0.5,
    save_dataset=               true,
    log_weighting=              true,
    samples_per_input=          1_000_000,
    variance_relative_epsilon=  1e-2,
    dirac_std=                  0.1,
    std_magnitude_factor=       200, # 114.0 is the max in TTT analytical
    implied_z_max_error=        50, # 38.0 is the max in TTT analytical
    uniform_quota=              0.2,
    variable_mean_dist=         Uniform(-100, 100),
    variable_std_dist=          Truncated(MixtureModel([Uniform(0,25), Uniform(25,600)], [.97,.03]), 1.0,600),
    factor_dist=                Distributions.Categorical([0.5,0.5])*2-3,
    bias_dist=                  Dirac(0),
    name_appendix=              "_TTTbased_stdmin_1-0"
)