include("../nn/mjl.jl")
using .NN: get_data

include("../../../lib/gaussian.jl")
using .GaussianDistribution
using ProgressMeter
using Format
using DelimitedFiles
using StatsBase

data_map = Dict{String, Any}()

ns = [1_000]#[10_000, 5_000, 2_000, 1_000, 500, 200, 100]
factors = [:wsf]#[:wsf, :gmf]

combinations = [(n, factor) for n in ns, factor in factors]
n_combinations = length(combinations)

progressbar = Progress(n_combinations, desc="Loading data...")
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
            samples_per_inputs = [1_000_000]#[10_000_000, 1_000_000, 100_000, 10_000, 1_000]
        else
            samples_per_inputs = [1_000_000]
        end

        for samples_per_input in samples_per_inputs
            nstring = replace(format(n, commas=true), "," => "_")
            filename = factor_path * "$(nstring)_$(samples_per_input)_Experiment_testfast.jld2"
            datapath = "SoHEstimation/approximate_message_passing/data/" * filename
            X, y =  get_data(datapath; target=target, factor=factor, transform_to_tau_rho=false)

            for i in 1:size(X, 1)
                if factor == :wsf && (X[i, 2] == 0 || X[i, 4] == 0 || X[i, 6] == 0 || y[i, 2] == 0)
                    error("Precision = 0 found!")
                end
            end

            key = "$(factor)_$(n)_$(samples_per_input)_$target"
            data_map[key] = (X, y)
        end
    end
    next!(progressbar)
end
finish!(progressbar)


# Calculate analytical marginal vs sampled marginal
function evaluate(inputs::Matrix{Float32}, labels::Matrix{Float32}, factor, n, samples_per_input, target; transform_to_tau_rho=false)

    metrics = Matrix{Float32}(undef, 24, size(inputs, 1))

    for i in 1:size(inputs, 1)
        input = inputs[i, :]
        label = labels[i, :]

        # Analytical solutions
        if factor == "wsf"
            if target == "targets_X"
                exact_marginal, exact_msg = NN.WeightedSumFactorGeneration.calc_msg_x(input, transform_to_tau_rho)
                if !transform_to_tau_rho
                    sampled_marginal = Gaussian1DFromMeanVariance(label...)
                    sampled_msg = sampled_marginal / Gaussian1DFromMeanVariance(NN.WeightedSumFactorGeneration.get_variable(NN.WeightedSumFactorGeneration.X, input)...)
                else
                    sampled_marginal = Gaussian1D(label...)
                    sampled_msg = sampled_marginal / Gaussian1D(NN.WeightedSumFactorGeneration.get_variable(NN.WeightedSumFactorGeneration.X, input)...)
                end
            elseif target == "targets_Y"
                exact_marginal, exact_msg = NN.WeightedSumFactorGeneration.calc_msg_y(input, transform_to_tau_rho)
                if !transform_to_tau_rho
                    sampled_marginal = Gaussian1DFromMeanVariance(label...)
                    sampled_msg = sampled_marginal / Gaussian1DFromMeanVariance(NN.WeightedSumFactorGeneration.get_variable(NN.WeightedSumFactorGeneration.Y, input)...)
                else
                    sampled_marginal = Gaussian1D(label...)
                    sampled_msg = sampled_marginal / Gaussian1D(NN.WeightedSumFactorGeneration.get_variable(NN.WeightedSumFactorGeneration.Y, input)...)
                end
            elseif target == "targets_Z"
                exact_marginal, exact_msg = NN.WeightedSumFactorGeneration.calc_msg_z(input, transform_to_tau_rho)
                if !transform_to_tau_rho
                    sampled_marginal = Gaussian1DFromMeanVariance(label...)
                    sampled_msg = sampled_marginal / Gaussian1DFromMeanVariance(NN.WeightedSumFactorGeneration.get_variable(NN.WeightedSumFactorGeneration.Z, input)...)
                else
                    sampled_marginal = Gaussian1D(label...)
                    sampled_msg = sampled_marginal / Gaussian1D(NN.WeightedSumFactorGeneration.get_variable(NN.WeightedSumFactorGeneration.Z, input)...)
                end
            end

        elseif factor == "gmf"
            if target == "targets_X"
                exact_marginal, exact_msg = NN.GaussianMeanFactorGeneration.calc_msg_x(input, transform_to_tau_rho)
                if !transform_to_tau_rho
                    sampled_marginal = Gaussian1DFromMeanVariance(label...)
                    sampled_msg = sampled_marginal / Gaussian1DFromMeanVariance(NN.GaussianMeanFactorGeneration.get_variable(NN.GaussianMeanFactorGeneration.X, input)...)
                else
                    sampled_marginal = Gaussian1D(label...)
                    sampled_msg = sampled_marginal / Gaussian1D(NN.GaussianMeanFactorGeneration.get_variable(NN.GaussianMeanFactorGeneration.X, input)...)
                end
            elseif target == "targets_Y"
                exact_marginal, exact_msg = NN.GaussianMeanFactorGeneration.calc_msg_y(input, transform_to_tau_rho)
                if !transform_to_tau_rho
                    sampled_marginal = Gaussian1DFromMeanVariance(label...)
                    msgBack = Gaussian1DFromMeanVariance(NN.GaussianMeanFactorGeneration.get_variable(NN.GaussianMeanFactorGeneration.Y, input)...)
                    sampled_msg = sampled_marginal / msgBack
                else
                    sampled_marginal = Gaussian1D(label...)
                    msgBack = Gaussian1D(NN.GaussianMeanFactorGeneration.get_variable(NN.GaussianMeanFactorGeneration.Y, input)...)
                    sampled_msg = sampled_marginal / msgBack
                end
            end
        
        end

        metric = GaussianDistribution.squared_diff(sampled_marginal, Gaussian1D(exact_marginal.tau, exact_marginal.rho))
        metrics[1, i] = metric[1]
        metrics[2, i] = metric[2]
        metrics[3, i] = metric[3]
        metrics[4, i] = metric[4]

        metric = GaussianDistribution.squared_diff(sampled_msg, Gaussian1D(exact_msg.tau, exact_msg.rho))
        metrics[5, i] = metric[1]
        metrics[6, i] = metric[2]
        metrics[7, i] = metric[3]
        metrics[8, i] = metric[4]

        metric = GaussianDistribution.absolute_error(sampled_marginal, Gaussian1D(exact_marginal.tau, exact_marginal.rho))
        metrics[9, i] = metric[1]
        metrics[10, i] = metric[2]
        metrics[11, i] = metric[3]
        metrics[12, i] = metric[4]

        metric = GaussianDistribution.absolute_error(sampled_msg, Gaussian1D(exact_msg.tau, exact_msg.rho))
        metrics[13, i] = metric[1]
        metrics[14, i] = metric[2]
        metrics[15, i] = metric[3]
        metrics[16, i] = metric[4]

        metric = GaussianDistribution.absolute_percentage_error(sampled_marginal, Gaussian1D(exact_marginal.tau, exact_marginal.rho))
        metrics[17, i] = metric[1]
        metrics[18, i] = metric[2]
        metrics[19, i] = metric[3]
        metrics[20, i] = metric[4]

        metric = GaussianDistribution.absolute_percentage_error(sampled_msg, Gaussian1D(exact_msg.tau, exact_msg.rho))
        metrics[21, i] = metric[1]
        metrics[22, i] = metric[2]
        metrics[23, i] = metric[3]
        metrics[24, i] = metric[4]
    end

    # calculate rmse on each metrics 1-8
    rmse_marginal_mean = sqrt.(StatsBase.mean(metrics[1, :]))
    rmse_marginal_variance = sqrt.(StatsBase.mean(metrics[2, :]))
    rmse_marginal_rho = sqrt.(StatsBase.mean(metrics[3, :]))
    rmse_marginal_tau = sqrt.(StatsBase.mean(metrics[4, :]))

    rmse_msg_mean = sqrt.(StatsBase.mean(metrics[5, :]))
    rmse_msg_variance = sqrt.(StatsBase.mean(metrics[6, :]))
    rmse_msg_rho = sqrt.(StatsBase.mean(metrics[7, :]))
    rmse_msg_tau = sqrt.(StatsBase.mean(metrics[8, :]))

    mae_marginal_mean = StatsBase.mean(metrics[9, :])
    mae_marginal_variance = StatsBase.mean(metrics[10, :])
    mae_marginal_rho = StatsBase.mean(metrics[11, :])
    mae_marginal_tau = StatsBase.mean(metrics[12, :])

    mae_msg_mean = StatsBase.mean(metrics[13, :])
    mae_msg_variance = StatsBase.mean(metrics[14, :])
    mae_msg_rho = StatsBase.mean(metrics[15, :])
    mae_msg_tau = StatsBase.mean(metrics[16, :])

    mape_marginal_mean = StatsBase.mean(metrics[17, :])
    mape_marginal_variance = StatsBase.mean(metrics[18, :])
    mape_marginal_rho = StatsBase.mean(metrics[19, :])
    mape_marginal_tau = StatsBase.mean(metrics[20, :])

    mape_msg_mean = StatsBase.mean(metrics[21, :])
    mape_msg_variance = StatsBase.mean(metrics[22, :])
    mape_msg_rho = StatsBase.mean(metrics[23, :])
    mape_msg_tau = StatsBase.mean(metrics[24, :])

    # write metrics to a matrix
    metrics = [rmse_marginal_mean, rmse_marginal_variance, rmse_marginal_rho, rmse_marginal_tau,
               rmse_msg_mean, rmse_msg_variance, rmse_msg_rho, rmse_msg_tau,
               mae_marginal_mean, mae_marginal_variance, mae_marginal_rho, mae_marginal_tau,
               mae_msg_mean, mae_msg_variance, mae_msg_rho, mae_msg_tau, mape_marginal_mean, mape_marginal_variance, mape_marginal_rho, mape_marginal_tau,
                mape_msg_mean, mape_msg_variance, mape_msg_rho, mape_msg_tau]

    descriptions = ["rmse_marginal_mean", "rmse_marginal_variance", "rmse_marginal_rho", "rmse_marginal_tau",
                    "rmse_msg_mean", "rmse_msg_variance", "rmse_msg_rho", "rmse_msg_tau",
                    "mae_marginal_mean", "mae_marginal_variance", "mae_marginal_rho", "mae_marginal_tau",
                    "mae_msg_mean", "mae_msg_variance", "mae_msg_rho", "mae_msg_tau",
                    "mape_marginal_mean", "mape_marginal_variance", "mape_marginal_rho", "mape_marginal_tau",
                    "mape_msg_mean", "mape_msg_variance", "mape_msg_rho", "mape_msg_tau"]

    metrics = hcat(descriptions, metrics)

    # write matrix to txt file
    filepath = "SoHEstimation/approximate_message_passing/evaluation/results/metrics_$(factor)_$(n)_$(samples_per_input)_$(target)_testfast.txt"
    writedlm(filepath, metrics)
end

# iterate through each key in data_map
progressbar = Progress(n_combinations, desc="Evaluating samle marginal vs. exact ...")
for (key, (X, y)) in data_map
    factor, n, samples_per_input, target, variable = split(key, "_")
    evaluate(X, y, factor, parse(Int, n), parse(Int, samples_per_input), target*"_$(variable)")
    next!(progressbar)
end
finish!(progressbar)
