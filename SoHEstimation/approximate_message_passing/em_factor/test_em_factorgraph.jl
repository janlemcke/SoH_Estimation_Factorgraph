include("em_factorgraph.jl")
using .SoHEstimation: add_noise
using DataFrames
using CSV
using StatsBase

filepath = "SoHEstimation/simulator/data/battery_pack.csv"
true_data = CSV.read(filepath, DataFrame)

amount_of_timesteps = length(true_data.SoH)
println("Amount of timesteps: ", amount_of_timesteps)
max_iterations = 200
multiple = false
separate_loop = false
sampling = false

# only use the first 10 rows of true_data
#true_data = true_data[1:amount_of_timesteps, :]

function ensure_committed()
	try
		status = read(`git status --porcelain`, String)
		if !isempty(status)
			run(`git add -A`)
			run(`git commit -m "experiment run smart stopping"`)
		end
		return strip(read(`git rev-parse HEAD`, String))
	catch e
		return "Error: $(e)"
	end
end

function evaluate(soh_list, dsoc_list, true_data, filepath, commit_hash, amount_of_timesteps, multiple, separate_loop, max_iterations, iterations, time_in_seconds, sampling, nn_index, percentage)

	metrics = Matrix{Float32}(undef, 12, amount_of_timesteps)

    # Filepath for predictions vs true values
    predictions_filepath = replace(filepath, ".csv" => "_predictions.csv")

    # Prepare DataFrame to store predictions
    df_predictions = DataFrame(
        Commit_Hash = commit_hash * " " * string(percentage) * " " * string(nn_index) * " " * string(multiple) * " " * string(separate_loop) * " " * string(sampling),
        Timestep = 1:amount_of_timesteps,
        True_SoH = [true_data.SoH[i] for i in 1:amount_of_timesteps],
        Predicted_SoH = [GaussianDistribution.mean(soh_list[i]) for i in 1:amount_of_timesteps],
        True_DSoC = [true_data.DSoC[i] for i in 1:amount_of_timesteps],
        Predicted_DSoC = [GaussianDistribution.mean(dsoc_list[i]) for i in 1:amount_of_timesteps]
    )

    # Write to CSV (append if exists)
    if isfile(predictions_filepath)
        CSV.write(predictions_filepath, df_predictions; append = true)
    else
        CSV.write(predictions_filepath, df_predictions)
    end

    for source in ["SoH", "DSOC"]
        for i in 1:amount_of_timesteps
            if source == "SoH"
                predicted_marginal = soh_list[i]
                exact_marginal = GaussianDistribution.Gaussian1DFromMeanVariance(true_data.SoH[i], 0.0001)
            else
                predicted_marginal = dsoc_list[i]
                exact_marginal = GaussianDistribution.Gaussian1DFromMeanVariance(true_data.DSoC[i], 0.0001)
            end
            metric = GaussianDistribution.squared_diff(predicted_marginal, GaussianDistribution.Gaussian1D(exact_marginal.tau, exact_marginal.rho))
            metrics[1, i] = metric[1]
            metrics[2, i] = metric[2]
            metrics[3, i] = metric[3]
            metrics[4, i] = metric[4]

            metric = GaussianDistribution.absolute_error(predicted_marginal, GaussianDistribution.Gaussian1D(exact_marginal.tau, exact_marginal.rho))
            metrics[5, i] = metric[1]
            metrics[6, i] = metric[2]
            metrics[7, i] = metric[3]
            metrics[8, i] = metric[4]

            metric = GaussianDistribution.absolute_percentage_error_soft_fail(predicted_marginal, GaussianDistribution.Gaussian1D(exact_marginal.tau, exact_marginal.rho))
            metrics[9, i] = metric[1]
            metrics[10, i] = metric[2]
            metrics[11, i] = metric[3]
            metrics[12, i] = metric[4]

        end

        rmse_marginal_prediction_exact_mean = sqrt(StatsBase.mean(metrics[1, :]))
        rmse_marginal_prediction_exact_variance = sqrt(StatsBase.mean(metrics[2, :]))
        rmse_marginal_prediction_exact_rho = sqrt(StatsBase.mean(metrics[3, :]))
        rmse_marginal_prediction_exact_tau = sqrt(StatsBase.mean(metrics[4, :]))

        abs_marginal_prediction_exact_mean = StatsBase.mean(metrics[5, :])
        abs_marginal_prediction_exact_variance = StatsBase.mean(metrics[6, :])
        abs_marginal_prediction_exact_rho = StatsBase.mean(metrics[7, :])
        abs_marginal_prediction_exact_tau = StatsBase.mean(metrics[8, :])

        mape_marginal_prediction_exact_mean = StatsBase.mean(metrics[9, :])
        mape_marginal_prediction_exact_variance = StatsBase.mean(metrics[10, :])
        mape_marginal_prediction_exact_rho = StatsBase.mean(metrics[11, :])
        mape_marginal_prediction_exact_tau = StatsBase.mean(metrics[12, :])

        df = DataFrame(
            Commit_Hash = commit_hash,
            Source = source,
            Amount_of_Timesteps = amount_of_timesteps,
            Multiple = multiple,
            Separate_Loop = separate_loop,
            Sampling = sampling,
            Max_Iterations = max_iterations,
            Iterations = iterations,
            Time_In_Seconds = time_in_seconds,
            NN_Index = nn_index,
            Percentage = percentage,
            RMSE_Marginal_Mean = rmse_marginal_prediction_exact_mean,
            RMSE_Marginal_Variance = rmse_marginal_prediction_exact_variance,
            RMSE_Marginal_Rho = rmse_marginal_prediction_exact_rho,
            RMSE_Marginal_Tau = rmse_marginal_prediction_exact_tau,
            MAE_Marginal_Mean = abs_marginal_prediction_exact_mean,
            MAE_Marginal_Variance = abs_marginal_prediction_exact_variance,
            MAE_Marginal_Rho = abs_marginal_prediction_exact_rho,
            MAE_Marginal_Tau = abs_marginal_prediction_exact_tau,
            MAPE_Marginal_Mean = mape_marginal_prediction_exact_mean,
            MAPE_Marginal_Variance = mape_marginal_prediction_exact_variance,
            MAPE_Marginal_Rho = mape_marginal_prediction_exact_rho,
            MAPE_Marginal_Tau = mape_marginal_prediction_exact_tau,
        )

        # Append results to CSV
        if isfile(filepath)
            CSV.write(filepath, df; append = true)
        else
            CSV.write(filepath, df)
        end
    end
end


commit_hash = ensure_committed()
#commit_hash = "Test"

function run_experiment(data, true_data)
    
    # percentage = 1.0
    # for i in 1:3
    #     start_time = time()
    #     soh_list, dsoc_list, iteration = SoHEstimation.start_convergence(data, max_iterations = max_iterations, multiple = multiple, separate_loop = separate_loop, sampling = sampling, nn_index=i)
    #     time_in_seconds = time() - start_time
    #     evaluate(soh_list, dsoc_list, true_data, "SoHEstimation/approximate_message_passing/evaluation/results/emf_graph.csv", commit_hash, amount_of_timesteps, multiple, separate_loop, max_iterations, iteration, time_in_seconds, sampling, i, percentage)
    # end

    #start_time = time()
    #soh_list, dsoc_list, iteration = SoHEstimation.start_convergence(data, max_iterations = max_iterations, multiple = true, separate_loop = separate_loop, sampling = sampling, nn_index=0)
    #time_in_seconds = time() - start_time
    #evaluate(soh_list, dsoc_list, true_data, "SoHEstimation/approximate_message_passing/evaluation/results/emf_graph.csv", commit_hash, amount_of_timesteps, true, separate_loop, max_iterations, iteration, time_in_seconds, sampling, 0, 1.0)

    for percentage in [0.5]
        data_copy = deepcopy(data)
        total_points = size(data_copy, 1)  # Total number of rows in data
        num_selected = round(Int, total_points * percentage)  # Number of rows to keep
        indices = round.(Int, range(1, total_points, length=num_selected))  # Select indices with equal spacing
        println("First ten indice: ", indices[1:10])
        experiment_data = data_copy[indices, :]  # Subsampled data
        for nn_index in 2:2
            if nn_index == 4
                multiple = true
            else
                multiple = false
            end
            start_time = time()
            soh_list, dsoc_list, iteration = SoHEstimation.start_convergence(experiment_data, max_iterations = max_iterations, multiple = multiple, separate_loop = separate_loop, sampling = sampling, nn_index=nn_index)
            time_in_seconds = time() - start_time
            evaluate(soh_list, dsoc_list, true_data, "SoHEstimation/approximate_message_passing/evaluation/results/emf_graph.csv", commit_hash, amount_of_timesteps, multiple, separate_loop, max_iterations, iteration, time_in_seconds, sampling, nn_index, percentage)
        end
    end
    
end

run_experiment(true_data, true_data)
