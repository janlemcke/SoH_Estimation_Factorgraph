include("../nn/mjl.jl")
using .NN: get_data

include("../../../lib/gaussian.jl")
using .GaussianDistribution
using ProgressMeter
using Format
using StatsBase
using CSV
using DataFrames

data_map = Dict{String, Any}()

ns = [100, 20_000, 10_000, 5_000, 2_000, 1_000, 500, 200]
factors = [:wsf, :gmf]

combinations = [(n, factor) for n in ns, factor in factors]
n_combinations = length(combinations)

filepath = "SoHEstimation/approximate_message_passing/evaluation/results/metrics_marginal_vs_exact_MA.csv"

# Delete old results file if it exists
if isfile(filepath)
    rm(filepath)
end

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

        if n == 5_000 || n == 10_000
            samples_per_inputs = [10_000_000, 1_000_000, 100_000, 10_000, 1_000]
        else
            samples_per_inputs = [1_000_000]
        end

        for samples_per_input in samples_per_inputs
            nstring = replace(format(n, commas=true), "," => "_")
            for i in 1:3
                filename = factor_path * "$(nstring)_$(samples_per_input)_Experiment_$i.jld2"
                datapath = "SoHEstimation/approximate_message_passing/data_masterarbeit/" * filename

                # Check if the file exists
                if !isfile(datapath)
                    continue
                end

                X, y =  get_data(datapath; target=target, factor=factor, transform_to_tau_rho=false)

                key = "$(factor)_$(n)_$(samples_per_input)_$(target)_$i"
                data_map[key] = (X, y)
            end
        end
    end
    next!(progressbar)
end
finish!(progressbar)

function ensure_committed()
    try
        status = read(`git status --porcelain`, String)
        if !isempty(status)
            run(`git add -A`)
            run(`git commit -m "experiment run"`)
        end
        return strip(read(`git rev-parse HEAD`, String))
    catch e
        return "Error: $(e)"
    end
end

commit_hash = ensure_committed()

function evaluate(inputs::Matrix{Float32}, labels::Matrix{Float32}, factor, n, samples_per_input, target, i; transform_to_tau_rho=false, filepath="")
    metrics = Matrix{Float32}(undef, 24, size(inputs, 1))

    for i in 1:size(inputs, 1)
        input = inputs[i, :]
        label = labels[i, :]

        # Analytical solutions
        if factor == "wsf"
            if target == "targets_X"
                exact_marginal, exact_msg = NN.WeightedSumFactorGeneration.calc_msg_x(input, transform_to_tau_rho)
                sampled_marginal, sampled_msg = compute_sampled_values(label, input, transform_to_tau_rho, NN.WeightedSumFactorGeneration, "X")
            elseif target == "targets_Y"
                exact_marginal, exact_msg = NN.WeightedSumFactorGeneration.calc_msg_y(input, transform_to_tau_rho)
                sampled_marginal, sampled_msg = compute_sampled_values(label, input, transform_to_tau_rho, NN.WeightedSumFactorGeneration, "Y")
            elseif target == "targets_Z"
                exact_marginal, exact_msg = NN.WeightedSumFactorGeneration.calc_msg_z(input, transform_to_tau_rho)
                sampled_marginal, sampled_msg = compute_sampled_values(label, input, transform_to_tau_rho, NN.WeightedSumFactorGeneration, "Z")
            end
        elseif factor == "gmf"
            if target == "targets_X"
                exact_marginal, exact_msg = NN.GaussianMeanFactorGeneration.calc_msg_x(input, transform_to_tau_rho)
                sampled_marginal, sampled_msg = compute_sampled_values(label, input, transform_to_tau_rho, NN.GaussianMeanFactorGeneration, "X")
            elseif target == "targets_Y"
                exact_marginal, exact_msg = NN.GaussianMeanFactorGeneration.calc_msg_y(input, transform_to_tau_rho)
                sampled_marginal, sampled_msg = compute_sampled_values(label, input, transform_to_tau_rho, NN.GaussianMeanFactorGeneration, "Y")
            end
        end

        # Compute different metric values
        compute_metrics!(metrics, i, sampled_marginal, sampled_msg, exact_marginal, exact_msg)
    end

    # Compute summary statistics (RMSE, MAE, MAPE)
    summary_metrics = compute_summary_metrics(metrics)

    # Convert metrics to DataFrame for CSV storage
    df = DataFrame(
        "Commit_Hash" => commit_hash,
        "Source" => "Sampled_vs_Exact",
        "Factor" => factor,
        "N" => n,
        "SamplesPerInput" => samples_per_input,
        "Experiment" => i,
        "Target" => target,
        "RMSE_Marginal_Mean" => summary_metrics[1],
        "RMSE_Marginal_Variance" => summary_metrics[2],
        "RMSE_Marginal_Rho" => summary_metrics[3],
        "RMSE_Marginal_Tau" => summary_metrics[4],
        "RMSE_Msg_Mean" => summary_metrics[5],
        "RMSE_Msg_Variance" => summary_metrics[6],
        "RMSE_Msg_Rho" => summary_metrics[7],
        "RMSE_Msg_Tau" => summary_metrics[8],
        "MAE_Marginal_Mean" => summary_metrics[9],
        "MAE_Marginal_Variance" => summary_metrics[10],
        "MAE_Marginal_Rho" => summary_metrics[11],
        "MAE_Marginal_Tau" => summary_metrics[12],
        "MAE_Msg_Mean" => summary_metrics[13],
        "MAE_Msg_Variance" => summary_metrics[14],
        "MAE_Msg_Rho" => summary_metrics[15],
        "MAE_Msg_Tau" => summary_metrics[16],
        "MAPE_Marginal_Mean" => summary_metrics[17],
        "MAPE_Marginal_Variance" => summary_metrics[18],
        "MAPE_Marginal_Rho" => summary_metrics[19],
        "MAPE_Marginal_Tau" => summary_metrics[20],
        "MAPE_Msg_Mean" => summary_metrics[21],
        "MAPE_Msg_Variance" => summary_metrics[22],
        "MAPE_Msg_Rho" => summary_metrics[23],
        "MAPE_Msg_Tau" => summary_metrics[24]
    )

    # Append results to CSV
    if isfile(filepath)
        CSV.write(filepath, df; append=true)
    else
        CSV.write(filepath, df)
    end
end

function compute_sampled_values(label, input, transform_to_tau_rho, factor_module , var)
    if !transform_to_tau_rho
        sampled_marginal = Gaussian1DFromMeanVariance(label...)
        sampled_msg = sampled_marginal / Gaussian1DFromMeanVariance(factor_module.get_variable(getfield(factor_module, Symbol(var)), input)...)
    else
        sampled_marginal = Gaussian1D(label...)
        sampled_msg = sampled_marginal / Gaussian1D(factor_module.get_variable(getfield(factor_module, Symbol(var)), input)...)
    end
    return sampled_marginal, sampled_msg
end

function compute_metrics!(metrics, i, sampled_marginal, sampled_msg, exact_marginal, exact_msg)
    metrics[1:4, i] .= GaussianDistribution.squared_diff(sampled_marginal, Gaussian1D(exact_marginal.tau, exact_marginal.rho))
    metrics[5:8, i] .= GaussianDistribution.squared_diff(sampled_msg, Gaussian1D(exact_msg.tau, exact_msg.rho))
    metrics[9:12, i] .= GaussianDistribution.absolute_error(sampled_marginal, Gaussian1D(exact_marginal.tau, exact_marginal.rho))
    metrics[13:16, i] .= GaussianDistribution.absolute_error(sampled_msg, Gaussian1D(exact_msg.tau, exact_msg.rho))
    metrics[17:20, i] .= GaussianDistribution.absolute_percentage_error(sampled_marginal, Gaussian1D(exact_marginal.tau, exact_marginal.rho))
    metrics[21:24, i] .= GaussianDistribution.absolute_percentage_error(sampled_msg, Gaussian1D(exact_msg.tau, exact_msg.rho))
end

function compute_summary_metrics(metrics)
    return [
        sqrt(StatsBase.mean(metrics[1, :])), sqrt(StatsBase.mean(metrics[2, :])),
        sqrt(StatsBase.mean(metrics[3, :])), sqrt(StatsBase.mean(metrics[4, :])),
        sqrt(StatsBase.mean(metrics[5, :])), sqrt(StatsBase.mean(metrics[6, :])),
        sqrt(StatsBase.mean(metrics[7, :])), sqrt(StatsBase.mean(metrics[8, :])),
        StatsBase.mean(metrics[9, :]), StatsBase.mean(metrics[10, :]),
        StatsBase.mean(metrics[11, :]), StatsBase.mean(metrics[12, :]),
        StatsBase.mean(metrics[13, :]), StatsBase.mean(metrics[14, :]),
        StatsBase.mean(metrics[15, :]), StatsBase.mean(metrics[16, :]),
        StatsBase.mean(metrics[17, :]), StatsBase.mean(metrics[18, :]),
        StatsBase.mean(metrics[19, :]), StatsBase.mean(metrics[20, :]),
        StatsBase.mean(metrics[21, :]), StatsBase.mean(metrics[22, :]),
        StatsBase.mean(metrics[23, :]), StatsBase.mean(metrics[24, :])
    ]
end


# iterate through each key in data_map
progressbar = Progress(n_combinations, desc="Evaluating sampled marginal vs. exact ...")
for (key, (X, y)) in data_map
    factor, n, samples_per_input, target, variable, i = split(key, "_")
    evaluate(X, y, factor, parse(Int, n), parse(Int, samples_per_input), target*"_$(variable)", i, filepath=filepath)
    next!(progressbar)
end
finish!(progressbar)