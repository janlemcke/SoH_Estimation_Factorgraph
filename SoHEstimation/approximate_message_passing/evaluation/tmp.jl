using JLD2
include("../../../lib/gaussian.jl")
using .GaussianDistribution
using Format


ns = [10_000, 5_000, 2_000, 1_000, 500, 200, 100]
factors = [:wsf, :gmf]

combinations = [(n, factor) for n in ns, factor in factors]
n_combinations = length(combinations)

for (n, factor) in combinations
    if factor ==:wsf
        targets = ["targets_X", "targets_Y", "targets_Z"]
    elseif factor == :gmf
        targets = ["targets_X", "targets_Y"]
    end
    for target in targets
        if factor == :wsf
            factor_path = "dataset_weighted_sum_factor_"
        else
            factor_path = "dataset_gaussian_mean_factor_"
        end

        if n == 5_000
            samples_per_inputs = [10_000_000, 1_000_000, 100_000, 10_000, 1_000]
        else
            samples_per_inputs = [1_000_000]
        end

        for samples_per_input in samples_per_inputs
            nstring = replace(format(n, commas=true), "," => "_")
            filename = factor_path * "$(nstring)_$(samples_per_input)_Experiment.jld2"
            datapath = "SoHEstimation/approximate_message_passing/data/" * filename
            println("Testing: $datapath")
            d = load(datapath)
            s = d["samples"]
            ty = d["targets_Y"]
            tx = d["targets_X"]
            for i in length(tx)
                msgtox = Gaussian1D(tx[i]...) / Gaussian1D(s[i][1],s[i][2])
                msgtoy = Gaussian1D(ty[i]...) / Gaussian1D(s[i][3],s[i][4])
                if msgtox.rho == 0 || msgtoy.rho == 0
                    throw("oh no: $datapath")
                end
            end
            println("all good")
        end
    end
    
end

