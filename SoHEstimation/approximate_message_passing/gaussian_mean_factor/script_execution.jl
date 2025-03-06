include("../../../lib/gaussian.jl")
include("generate_data_gaussian_mean_factor.jl")

using Distributions
using .GaussianMeanFactorGeneration: generate_dataset_gaussian_mean_factor

std_magnitude_factor = 10

ns = [100, 20_000, 10_000, 5_000, 2_000, 1_000, 500, 200]
samples_per_inputs = [1_000_000]

for n in ns
    for i in 1:3
        for samples_per_input in samples_per_inputs
            generate_dataset_gaussian_mean_factor(
                n=n, save_dataset=true, patience=2,
                log_weighting=true,
                samples_per_input=samples_per_input,
                variance_relative_epsilon=1e-2,
                variable_mean_dist=Uniform(-100, 100),
                variable_std_dist= Product([Uniform(1,20) for _ in 1:2]),#MixtureModel([Product([Uniform(0,20) for _ in 1:2]), Product([Uniform(20,200) for _ in 1:2]), Product([Uniform(0,20), Uniform(20, 200)])],[0.7 ,0.15, 0.15]),
                factor_dist=Uniform(1, 10),
                std_magnitude_factor=std_magnitude_factor,
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
            generate_dataset_gaussian_mean_factor(
                n=n, save_dataset=true, patience=2,
                log_weighting=true,
                samples_per_input=samples_per_input,
                variance_relative_epsilon=1e-2,
                variable_mean_dist=Uniform(-100, 100),
                variable_std_dist= Product([Uniform(1,20) for _ in 1:2]),#MixtureModel([Product([Uniform(0,20) for _ in 1:2]), Product([Uniform(20,200) for _ in 1:2]), Product([Uniform(0,20), Uniform(20, 200)])],[0.7 ,0.15, 0.15]),
                factor_dist=Uniform(1, 10),
                std_magnitude_factor=std_magnitude_factor,
                name_appendix="_$(samples_per_input)_Experiment_$i",
                savepath="SoHEstimation/approximate_message_passing/data_masterarbeit/",
                strict_convergence=false
            )
        end
    end
end
