include("../../../lib/gaussian.jl")
# include("../../../lib/utils.jl")
include("generate_data_weighted_sum_factor.jl")
include("plot_nn_results.jl")

using Distributions

ns = [100, 20_000, 10_000, 5_000, 2_000, 1_000, 500, 200]
samples_per_inputs = [1_000_000]
for n in ns
    for i in 1:3
        for samples_per_input in samples_per_inputs
            dataset = WeightedSumFactorGeneration.generate_dataset_weighted_sum_factor(
                n=                          n,
                save_dataset=               true,
                log_weighting=              true,
                samples_per_input=          samples_per_input,
                variance_relative_epsilon=  1e-2,
                dirac_std=                  0.1,
                std_magnitude_factor=       10, # 114.0 is the max in TTT analytical
                uniform_quota=              0.1,
                variable_mean_dist=         Uniform(-100, 100),
                variable_std_dist=          Product([Uniform(1,20) for _ in 1:3]),#MixtureModel([Product([Uniform(0,20) for _ in 1:3]), Product([Uniform(20,200) for _ in 1:3]), Product([Uniform(0,20), Uniform(0,20), Uniform(20, 200)]), Product([Uniform(0,20), Uniform(20,200), Uniform(20, 200)])],[0.7 ,0.1, 0.1, 0.1]),
                factor_dist=                Uniform(-10, 10), # Hier versuchen nur Ints zu samplen.
                bias_dist=                  Dirac(0),
                algorithm=                  :adaptive_metropolis_hastings, 
                name_appendix="_$(samples_per_input)_Experiment_$i",
                savepath="SoHEstimation/approximate_message_passing/data_masterarbeit/",
                strict_convergence=false
            )
        end
    end
end

ns = [10_000, 5_000]
samples_per_inputs = [1_000, 10_000, 100_000, 10_000_000]
for n in ns
    for i in 1:3
        for samples_per_input in samples_per_inputs
            dataset = WeightedSumFactorGeneration.generate_dataset_weighted_sum_factor(
                n=                          n,
                save_dataset=               true,
                log_weighting=              true,
                samples_per_input=          samples_per_input,
                variance_relative_epsilon=  1e-2,
                dirac_std=                  0.1,
                std_magnitude_factor=       10, # 114.0 is the max in TTT analytical
                uniform_quota=              0.1,
                variable_mean_dist=         Uniform(-100, 100),
                variable_std_dist=          Product([Uniform(1,20) for _ in 1:3]),#MixtureModel([Product([Uniform(0,20) for _ in 1:3]), Product([Uniform(20,200) for _ in 1:3]), Product([Uniform(0,20), Uniform(0,20), Uniform(20, 200)]), Product([Uniform(0,20), Uniform(20,200), Uniform(20, 200)])],[0.7 ,0.1, 0.1, 0.1]),
                factor_dist=                Uniform(-10, 10), # Hier versuchen nur Ints zu samplen.
                bias_dist=                  Dirac(0),
                algorithm=                  :adaptive_metropolis_hastings, 
                name_appendix="_$(samples_per_input)_Experiment_$i",
                savepath="SoHEstimation/approximate_message_passing/data_masterarbeit/",
                strict_convergence=false
            )
        end
    end
end
