include("../../../lib/gaussian.jl")
include("generate_dataset_em_factor.jl")

using Distributions



ns = [10_000]#][100, 20_000, 10_000, 5_000, 2_000, 1_000, 500, 200]
samples_per_inputs = [1_000_000]
for n in ns
    for i in 1:3
        for samples_per_input in samples_per_inputs
            ElectricalModelFactorGeneration.generate_dataset_em_factor(
                n=                          n,
                save_dataset=               true,
                log_weighting=              true,
                samples_per_input=          samples_per_input,
                variance_relative_epsilon=  1e-2,
                std_magnitude_factor=       3,
                uniform_quota=              0.05,
                I_mean_dist=                Uniform(-11, 11), 
                I_std_dist=                 Uniform(0.05, 0.2),
                SOH_mean_dist=              Uniform(0.68, 0.7),
                SOH_std_dist=               Uniform(0.02, 0.05),
                DSOC_mean_dist=             Uniform(-0.1, 0.1),
                DSOC_std_dist=              Uniform(0.03, 0.15),
                qo_dist=                    Uniform(65.5, 66.5),
                dt_dist=                    Uniform(0.0040, 0.0042),
                implied_z_max_error=        0.005,
                savepath=                   "SoHEstimation/approximate_message_passing/data_masterarbeit/",
                name_appendix=              "_$(samples_per_input)_Experiment_$(i)",
                strict_convergence=false
            )                  
        end
    end
end

"""

ns = [10_000, 5_000]
samples_per_inputs = [1_000, 10_000, 100_000, 10_000_000]
for n in ns
    for i in 1:3
        for samples_per_input in samples_per_inputs
            ElectricalModelFactorGeneration.generate_dataset_em_factor(
                n=                          n,
                save_dataset=               true,
                log_weighting=              true,
                samples_per_input=          1_000_000,
                variance_relative_epsilon=  1e-2,
                std_magnitude_factor=       10,
                uniform_quota=              0.05,
                I_mean_dist=                Uniform(-100, 100),
                I_std_dist=                 Uniform(1e-3, 8.0),
                SOH_mean_dist=              Uniform(0.1, 1.0),
                SOH_std_dist=               Uniform(1e-3, 0.25),
                qo_dist=                    Uniform(30, 120.0),
                dt_dist=                    Uniform(1e-4, 10.0),
                implied_z_max_error=        50,
                savepath=                   "SoHEstimation/approximate_message_passing/em_factor/data_masterarbeit",
                name_appendix=              "_$(samples_per_input)_Experiment_$i",
                strict_convergence=false
            )
        end
    end
end

"""