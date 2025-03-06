include("../nn/mjl.jl")
using .NN: predict_sample, load_model
using ProgressMeter
using Random
using Base.Threads
using DelimitedFiles
using Distributions
using CUDA
using CSV
using DataFrames
using StatsBase

include("../../../lib/gaussian.jl")
using .GaussianDistribution: Gaussian1D, Gaussian1DFromMeanVariance

include("../weighted_sum_factor/generate_data_weighted_sum_factor.jl")
using .WeightedSumFactorGeneration: generate_output_weighted_sum_factor

include("../gaussian_mean_factor/generate_data_gaussian_mean_factor.jl")
using .GaussianMeanFactorGeneration: generate_output_gaussian_mean_factor

factors = ["wsf", "gmf"]
filepath = "SoHEstimation/approximate_message_passing/evaluation/results/metrics_fixed_large.csv"
variance_relative_epsilon = 1e-2
number_of_sampels_per_input = 3

gridpoints = []

for gp_mean in -100:5:100
	for gp_std in 1:1:20
		gp_var = gp_std^2
		push!(gridpoints, [gp_mean, gp_var])
	end
end
# transform gridpoints to a 2xn matrix
gridpoints = hcat(gridpoints...)
println("Shape of gridpoints: ", size(gridpoints))


n_points = size(gridpoints, 2)

gridpoints_constants = []
for gp_a in -10:10
	for gp_b in -10:10
		if gp_a == 0 || gp_b == 0
			continue
		end
		push!(gridpoints_constants, [gp_a, gp_b])
	end
end

# transform gridpoints to a 2xn matrix
gridpoints_constants = hcat(gridpoints_constants...)

n_points_constants = size(gridpoints_constants, 2)

gridpoints_beta = []
for gp_beta in 1:10
	push!(gridpoints_beta, gp_beta)
end

gridpoints_beta = hcat(gridpoints_beta...)
n_points_beta = length(gridpoints_beta)


function ensure_committed()
	try
		status = read(`git status --porcelain`, String)
		if !isempty(status)
			run(`git add -A`)
			run(`git commit -m "experiment 3 run"`)
		end
		return strip(read(`git rev-parse HEAD`, String))
	catch e
		return "Error: $(e)"
	end
end

commit_hash = ensure_committed()
#commit_hash = "test"

# LOAD NEURAL networks
path = "SoHEstimation/approximate_message_passing/evaluation/models/"
files = readdir(path)
nn_map = Dict{String, Any}()


wsf_models_x = []
wsf_models_y = []
wsf_models_z = []

gmf_models_x = []
gmf_models_y = []

for file in files
	# filter out jls files
	if occursin(".jls", file) && occursin("gmf", file)
		# load the file
		nn = load_model(path * file)
		if occursin("X", file)
			push!(gmf_models_x, nn)
		elseif occursin("Y", file)
			push!(gmf_models_y, nn)
		end
	end

	if occursin(".jls", file) && occursin("wsf", file)
		# load the file
		nn = load_model(path * file)
		if occursin("X", file)
			push!(wsf_models_x, nn)
		elseif occursin("Y", file)
			push!(wsf_models_y, nn)
		elseif occursin("Z", file)
			push!(wsf_models_z, nn)
		end
	end
end


nn_map["wsf"] = Dict("targets_X" => wsf_models_x, "targets_Y" => wsf_models_y, "targets_Z" => wsf_models_z)
nn_map["gmf"] = Dict("targets_X" => gmf_models_x, "targets_Y" => gmf_models_y)


println("Loaded models.")



function get_analytical_output(inputs, variable, f, gridpoints)
	analytical_marginals = Matrix{Gaussian1D}(undef, 1, size(gridpoints, 2))
	analytical_msgs = Matrix{Gaussian1D}(undef, 1, size(gridpoints, 2))
	
	for j in 1:size(gridpoints, 2)
		input = inputs[:, j]
		exact_marginal, exact_msg = f(input, false)

		if !isfinite(exact_marginal.rho) || !isfinite(exact_marginal.tau)
			error("Exact marginal $exact_marginal is not finite given input: $input")
		end

		if !isfinite(exact_msg.rho) || !isfinite(exact_msg.tau)
			error("Exact message $exact_msg is nan or inf given input: $input")
		end

		analytical_marginals[1, j] = Gaussian1D(exact_marginal.tau, exact_marginal.rho)
		analytical_msgs[1, j] = Gaussian1D(exact_msg.tau, exact_msg.rho)
	end

	return analytical_marginals, analytical_msgs
end

function get_sampled_output_wsf(inputs, variable, factor, gridpoints)
	println("Sampling outputs for variable $variable and factor $factor ...")
	output_marginals = Matrix{Gaussian1D}(undef, number_of_sampels_per_input, size(gridpoints, 2))
	output_msgs = Matrix{Gaussian1D}(undef, number_of_sampels_per_input, size(gridpoints, 2))

	progressbar = Progress(size(gridpoints, 2), desc = "Sampling outputs for variable $variable and factor $factor and $(size(gridpoints, 2)) gridpoints ...")

	Threads.@threads for j in 1:size(gridpoints, 2)
		current_input = inputs[:, j]
		for i in 1:number_of_sampels_per_input
			while true
				try
					sampled_output, _, _ = generate_output_weighted_sum_factor(current_input,
						samples_per_input = 1_000_000,
						log_weighting = true,
						algorithm = :adaptive_metropolis_hastings,
						set_large_variances_to_uniform = false,
						variance_relative_epsilon = variance_relative_epsilon,
						dirac_std = 0.1,
						debug = false,
					)
					mean, var = WeightedSumFactorGeneration.get_variable(variable, sampled_output)
					sampled_marginal = Gaussian1DFromMeanVariance(mean, var)

					if !isfinite(sampled_marginal.rho) || !isfinite(sampled_marginal.tau)
						error("Sampled marginal $sampled_marginal is nan or inf given input: $current_input")
					end

					sampled_msg = sampled_marginal / Gaussian1DFromMeanVariance(WeightedSumFactorGeneration.get_variable(variable, current_input)...)

					if !isfinite(sampled_msg.rho) || !isfinite(sampled_msg.tau)
						error("Sampled message $sampled_msg is nan or inf given input: $current_input")
					end
					output_marginals[i, j] = sampled_marginal
					output_msgs[i, j] = sampled_msg
					break
				catch e
					#println("Error during sampling: $e. Retrying...")
				end
			end
		end
		next!(progressbar)
	end

	return output_marginals, output_msgs
end

function get_sampled_output_gmf(inputs, variable, factor, gridpoints)
	output_marginals = Matrix{Gaussian1D}(undef, number_of_sampels_per_input, size(gridpoints, 2))
	output_msgs = Matrix{Gaussian1D}(undef, number_of_sampels_per_input, size(gridpoints, 2))

	progressbar = Progress(size(gridpoints, 2), desc = "Sampling outputs for variable $variable and factor $factor ...")

	Threads.@threads for j in 1:size(gridpoints, 2)
		current_input = inputs[:, j]
		for i in 1:number_of_sampels_per_input
			while true
				try
					sampled_output, _, _ = generate_output_gaussian_mean_factor(current_input,
						samples_per_input = 1_000_000,
						log_weighting = true,
						variance_relative_epsilon = variance_relative_epsilon,
					)
					mean, var = GaussianMeanFactorGeneration.get_variable(variable, sampled_output)
					sampled_marginal = Gaussian1DFromMeanVariance(mean, var)

					if !isfinite(sampled_marginal.rho) || !isfinite(sampled_marginal.tau)
						error("Sampled marginal $sampled_marginal is nan or inf given input: $current_input")
					end

					sampled_msg = sampled_marginal / Gaussian1DFromMeanVariance(GaussianMeanFactorGeneration.get_variable(variable, current_input)...)

					if !isfinite(sampled_msg.rho) || !isfinite(sampled_msg.tau)
						error("Sampled message $sampled_msg is nan or inf given input: $current_input")
					end
					output_marginals[i, j] = sampled_marginal
					output_msgs[i, j] = sampled_msg
					break
				catch e
					#println("Error during sampling: $e. Retrying...")
				end
			end
		end
		next!(progressbar)
	end

	return output_marginals, output_msgs
end


function get_predicted_output_wsf(inputs, models, variable, gridpoints)
	predicted_marginals = Matrix{Gaussian1D}(undef, number_of_sampels_per_input, size(gridpoints, 2))
	predicted_msgs = Matrix{Gaussian1D}(undef, number_of_sampels_per_input, size(gridpoints, 2))
	for j in 1:size(gridpoints, 2)
		current_input = inputs[:, j]

		for i in 1:number_of_sampels_per_input
			prediction = predict_sample(models[i], current_input)
			if CUDA.functional()
				prediction = Matrix{Float32}(prediction)
			end
			# check if prediction [1] is nan or inf
			if any(isnan.(prediction)) || any(isinf.(prediction))
				error("Prediction $prediction is nan or inf given input: $current_input")
			end
			predicted_marginal = Gaussian1DFromMeanVariance(prediction[1], prediction[2])
			predicted_msg = predicted_marginal / Gaussian1DFromMeanVariance(WeightedSumFactorGeneration.get_variable(variable, current_input)...)

			if !isfinite(predicted_msg.rho) || !isfinite(predicted_msg.tau)
				error("predicted message $predicted_msg is nan or inf given input: $current_input")
			end

			predicted_marginals[i, j] = predicted_marginal
			predicted_msgs[i, j] = predicted_msg
		end
	end

	return predicted_marginals, predicted_msgs
end

function get_predicted_output_gmf(inputs, models, variable, gridpoints)
	predicted_marginals = Matrix{Gaussian1D}(undef, number_of_sampels_per_input, size(gridpoints, 2))
	predicted_msgs = Matrix{Gaussian1D}(undef, number_of_sampels_per_input, size(gridpoints, 2))

	for j in 1:size(gridpoints, 2)
		current_input = inputs[:, j]

		for i in 1:number_of_sampels_per_input
			prediction = predict_sample(models[i], current_input)
			if CUDA.functional()
				prediction = Matrix{Float32}(prediction)
			end
			# check if prediction [1] is nan or inf
			if any(isnan.(prediction)) || any(isinf.(prediction))
				error("Prediction $prediction is nan or inf given input: $current_input")
			end
			predicted_marginal = Gaussian1DFromMeanVariance(prediction[1], prediction[2])
			predicted_msg = predicted_marginal / Gaussian1DFromMeanVariance(GaussianMeanFactorGeneration.get_variable(variable, current_input)...)

			if !isfinite(predicted_msg.rho) || !isfinite(predicted_msg.tau)
				error("predicted message $predicted_msg is nan or inf given input: $current_input")
			end

			predicted_marginals[i, j] = predicted_marginal
			predicted_msgs[i, j] = predicted_msg
		end
	end

	return predicted_marginals, predicted_msgs
end

function calc_metrics(nn_output, sampled_output, analytical_output, gridpoints)

	nn_marginals, nn_messages = nn_output
	sampled_marginals, sampled_messages = sampled_output
	analytical_marginals, analytical_messages = analytical_output

	nn_vs_exact_marginal_rmse = []
	nn_vs_exact_message_rmse = []
	nn_vs_exact_marginal_mae = []
	nn_vs_exact_message_mae = []
	nn_vs_exact_marginal_mape = []
	nn_vs_exact_message_mape = []

	nn_vs_sampled_marginal_rmse = []
	nn_vs_sampled_message_rmse = []
	nn_vs_sampled_marginal_mae = []
	nn_vs_sampled_message_mae = []
	nn_vs_sampled_marginal_mape = []
	nn_vs_sampled_message_mape = []

	sampled_vs_exact_marginal_rmse = []
	sampled_vs_exact_message_rmse = []
	sampled_vs_exact_marginal_mae = []
	sampled_vs_exact_message_mae = []
	sampled_vs_exact_marginal_mape = []
	sampled_vs_exact_message_mape = []

	# Evaluate nn vs analytical
	for i in 1:size(gridpoints, 2)
		nn_marginals_for_gridpoint = nn_marginals[:, i]
		nn_messages_for_gridpoint = nn_messages[:, i]

		sampled_marginals_for_gridpoint = sampled_marginals[:, i]
		sampled_messages_for_gridpoint = sampled_messages[:, i]

		exact_marginal = analytical_marginals[i]
		exact_message = analytical_messages[i]

		nn_vs_exact_marginal_errors_squared_diff = []
		nn_vs_exact_message_errors_squared_diff = []
		nn_vs_exact_marginal_errors_absolute = []
		nn_vs_exact_message_errors_absolute = []
		nn_vs_exact_marginal_errors_absolute_percentage = []
		nn_vs_exact_message_errors_absolute_percentage = []

		nn_vs_sampled_marginal_errors_squared_diff = []
		nn_vs_sampled_message_errors_squared_diff = []
		nn_vs_sampled_marginal_errors_absolute = []
		nn_vs_sampled_message_errors_absolute = []
		nn_vs_sampled_marginal_errors_absolute_percentage = []
		nn_vs_sampled_message_errors_absolute_percentage = []

		sampled_vs_exact_marginal_errors_squared_diff = []
		sampled_vs_exact_message_errors_squared_diff = []
		sampled_vs_exact_marginal_errors_absolute = []
		sampled_vs_exact_message_errors_absolute = []
		sampled_vs_exact_marginal_errors_absolute_percentage = []
		sampled_vs_exact_message_errors_absolute_percentage = []

		for j in 1:number_of_sampels_per_input
			nn_marginal = nn_marginals_for_gridpoint[j]
			nn_message = nn_messages_for_gridpoint[j]
			sampled_marginal = sampled_marginals_for_gridpoint[j]
			sampled_message = sampled_messages_for_gridpoint[j]

			# nn vs. exact
			push!(nn_vs_exact_marginal_errors_squared_diff, [sqrt.(GaussianDistribution.squared_diff(nn_marginal, exact_marginal))...])
			push!(nn_vs_exact_marginal_errors_absolute, [GaussianDistribution.absolute_error(nn_marginal, exact_marginal)...])
			push!(nn_vs_exact_marginal_errors_absolute_percentage, [GaussianDistribution.absolute_percentage_error_soft_fail(nn_marginal, exact_marginal)...])

			push!(nn_vs_exact_message_errors_squared_diff, [sqrt.(GaussianDistribution.squared_diff(nn_message, exact_message))...])
			push!(nn_vs_exact_message_errors_absolute, [GaussianDistribution.absolute_error(nn_message, exact_message)...])
			push!(nn_vs_exact_message_errors_absolute_percentage, [GaussianDistribution.absolute_percentage_error_soft_fail(nn_message, exact_message)...])

			# nn vs. sampled
			push!(nn_vs_sampled_marginal_errors_squared_diff, [sqrt.(GaussianDistribution.squared_diff(nn_marginal, sampled_marginal))...])
			push!(nn_vs_sampled_marginal_errors_absolute, [GaussianDistribution.absolute_error(nn_marginal, sampled_marginal)...])
			push!(nn_vs_sampled_marginal_errors_absolute_percentage, [GaussianDistribution.absolute_percentage_error_soft_fail(nn_marginal, sampled_marginal)...])

			push!(nn_vs_sampled_message_errors_squared_diff, [sqrt.(GaussianDistribution.squared_diff(nn_message, sampled_message))...])
			push!(nn_vs_sampled_message_errors_absolute, [GaussianDistribution.absolute_error(nn_message, sampled_message)...])
			push!(nn_vs_sampled_message_errors_absolute_percentage, [GaussianDistribution.absolute_percentage_error_soft_fail(nn_message, sampled_message)...])

			# sampled vs. exact
			push!(sampled_vs_exact_marginal_errors_squared_diff, [sqrt.(GaussianDistribution.squared_diff(sampled_marginal, exact_marginal))...])
			push!(sampled_vs_exact_marginal_errors_absolute, [GaussianDistribution.absolute_error(sampled_marginal, exact_marginal)...])
			push!(sampled_vs_exact_marginal_errors_absolute_percentage, [GaussianDistribution.absolute_percentage_error_soft_fail(sampled_marginal, exact_marginal)...])

			push!(sampled_vs_exact_message_errors_squared_diff, [sqrt.(GaussianDistribution.squared_diff(sampled_message, exact_message))...])
			push!(sampled_vs_exact_message_errors_absolute, [GaussianDistribution.absolute_error(sampled_message, exact_message)...])
			push!(sampled_vs_exact_message_errors_absolute_percentage, [GaussianDistribution.absolute_percentage_error_soft_fail(sampled_message, exact_message)...])
		end

		nn_vs_exact_marginal_rmse_mean, nn_vs_exact_marginal_rmse_std = StatsBase.mean(nn_vs_exact_marginal_errors_squared_diff), StatsBase.std(nn_vs_exact_marginal_errors_squared_diff)
		nn_vs_exact_message_rmse_mean, nn_vs_exact_message_rmse_std = StatsBase.mean(nn_vs_exact_message_errors_squared_diff), StatsBase.std(nn_vs_exact_message_errors_squared_diff)
		nn_vs_exact_marginal_mae_mean, nn_vs_exact_marginal_mae_std = StatsBase.mean(nn_vs_exact_marginal_errors_absolute), StatsBase.std(nn_vs_exact_marginal_errors_absolute)
		nn_vs_exact_message_mae_mean, nn_vs_exact_message_mae_std = StatsBase.mean(nn_vs_exact_message_errors_absolute), StatsBase.std(nn_vs_exact_message_errors_absolute)

		nn_vs_sampled_marginal_rmse_mean, nn_vs_sampled_marginal_rmse_std = StatsBase.mean(nn_vs_sampled_marginal_errors_squared_diff), StatsBase.std(nn_vs_sampled_marginal_errors_squared_diff)
		nn_vs_sampled_message_rmse_mean, nn_vs_sampled_message_rmse_std = StatsBase.mean(nn_vs_sampled_message_errors_squared_diff), StatsBase.std(nn_vs_sampled_message_errors_squared_diff)
		nn_vs_sampled_marginal_mae_mean, nn_vs_sampled_marginal_mae_std = StatsBase.mean(nn_vs_sampled_marginal_errors_absolute), StatsBase.std(nn_vs_sampled_marginal_errors_absolute)
		nn_vs_sampled_message_mae_mean, nn_vs_sampled_message_mae_std = StatsBase.mean(nn_vs_sampled_message_errors_absolute), StatsBase.std(nn_vs_sampled_message_errors_absolute)

		sampled_vs_exact_marginal_rmse_mean, sampled_vs_exact_marginal_rmse_std = StatsBase.mean(sampled_vs_exact_marginal_errors_squared_diff), StatsBase.std(sampled_vs_exact_marginal_errors_squared_diff)
		sampled_vs_exact_message_rmse_mean, sampled_vs_exact_message_rmse_std = StatsBase.mean(sampled_vs_exact_message_errors_squared_diff), StatsBase.std(sampled_vs_exact_message_errors_squared_diff)
		sampled_vs_exact_marginal_mae_mean, sampled_vs_exact_marginal_mae_std = StatsBase.mean(sampled_vs_exact_marginal_errors_absolute), StatsBase.std(sampled_vs_exact_marginal_errors_absolute)
		sampled_vs_exact_message_mae_mean, sampled_vs_exact_message_mae_std = StatsBase.mean(sampled_vs_exact_message_errors_absolute), StatsBase.std(sampled_vs_exact_message_errors_absolute)

		nn_vs_exact_marginal_mape_mean, nn_vs_exact_marginal_mape_std = StatsBase.mean(nn_vs_exact_marginal_errors_absolute_percentage), StatsBase.std(nn_vs_exact_marginal_errors_absolute_percentage)
		nn_vs_sampled_marginal_mape_mean, nn_vs_sampled_marginal_mape_std = StatsBase.mean(nn_vs_sampled_marginal_errors_absolute_percentage), StatsBase.std(nn_vs_sampled_marginal_errors_absolute_percentage)
		sampled_vs_exact_marginal_mape_mean, sampled_vs_exact_marginal_mape_std = StatsBase.mean(sampled_vs_exact_marginal_errors_absolute_percentage), StatsBase.std(sampled_vs_exact_marginal_errors_absolute_percentage)


		nn_vs_exact_message_mape_mean, nn_vs_exact_message_mape_std = StatsBase.mean(nn_vs_exact_message_errors_absolute_percentage), StatsBase.std(nn_vs_exact_message_errors_absolute_percentage)
		nn_vs_sampled_message_mape_mean, nn_vs_sampled_message_mape_std = StatsBase.mean(nn_vs_sampled_message_errors_absolute_percentage), StatsBase.std(nn_vs_sampled_message_errors_absolute_percentage)
		sampled_vs_exact_message_mape_mean, sampled_vs_exact_message_mape_std = StatsBase.mean(sampled_vs_exact_message_errors_absolute_percentage), StatsBase.std(sampled_vs_exact_message_errors_absolute_percentage)

		push!(nn_vs_exact_marginal_rmse, [nn_vs_exact_marginal_rmse_mean, nn_vs_exact_marginal_rmse_std])
		push!(nn_vs_exact_message_rmse, [nn_vs_exact_message_rmse_mean, nn_vs_exact_message_rmse_std])
		push!(nn_vs_exact_marginal_mae, [nn_vs_exact_marginal_mae_mean, nn_vs_exact_marginal_mae_std])
		push!(nn_vs_exact_message_mae, [nn_vs_exact_message_mae_mean, nn_vs_exact_message_mae_std])
		push!(nn_vs_exact_marginal_mape, [nn_vs_exact_marginal_mape_mean, nn_vs_exact_marginal_mape_std])
		push!(nn_vs_exact_message_mape, [nn_vs_exact_message_mape_mean, nn_vs_exact_message_mape_std])

		push!(nn_vs_sampled_marginal_rmse, [nn_vs_sampled_marginal_rmse_mean, nn_vs_sampled_marginal_rmse_std])
		push!(nn_vs_sampled_message_rmse, [nn_vs_sampled_message_rmse_mean, nn_vs_sampled_message_rmse_std])
		push!(nn_vs_sampled_marginal_mae, [nn_vs_sampled_marginal_mae_mean, nn_vs_sampled_marginal_mae_std])
		push!(nn_vs_sampled_message_mae, [nn_vs_sampled_message_mae_mean, nn_vs_sampled_message_mae_std])
		push!(nn_vs_sampled_marginal_mape, [nn_vs_sampled_marginal_mape_mean, nn_vs_sampled_marginal_mape_std])
		push!(nn_vs_sampled_message_mape, [nn_vs_sampled_message_mape_mean, nn_vs_sampled_message_mape_std])

		push!(sampled_vs_exact_marginal_rmse, [sampled_vs_exact_marginal_rmse_mean, sampled_vs_exact_marginal_rmse_std])
		push!(sampled_vs_exact_message_rmse, [sampled_vs_exact_message_rmse_mean, sampled_vs_exact_message_rmse_std])
		push!(sampled_vs_exact_marginal_mae, [sampled_vs_exact_marginal_mae_mean, sampled_vs_exact_marginal_mae_std])
		push!(sampled_vs_exact_message_mae, [sampled_vs_exact_message_mae_mean, sampled_vs_exact_message_mae_std])
		push!(sampled_vs_exact_marginal_mape, [sampled_vs_exact_marginal_mape_mean, sampled_vs_exact_marginal_mape_std])
		push!(sampled_vs_exact_message_mape, [sampled_vs_exact_message_mape_mean, sampled_vs_exact_message_mape_std])

	end

	println("Size of nn_vs_exact_marginal_rmse: ", size(nn_vs_exact_marginal_rmse))

	# return metrics as a Dict
	return Dict(
		:nn_vs_exact_marginal_rmse => nn_vs_exact_marginal_rmse,
		:nn_vs_exact_message_rmse => nn_vs_exact_message_rmse,
		:nn_vs_exact_marginal_mae => nn_vs_exact_marginal_mae,
		:nn_vs_exact_message_mae => nn_vs_exact_message_mae,
		:nn_vs_exact_marginal_mape => nn_vs_exact_marginal_mape,
		:nn_vs_exact_message_mape => nn_vs_exact_message_mape,
		:nn_vs_sampled_marginal_rmse => nn_vs_sampled_marginal_rmse,
		:nn_vs_sampled_message_rmse => nn_vs_sampled_message_rmse,
		:nn_vs_sampled_marginal_mae => nn_vs_sampled_marginal_mae,
		:nn_vs_sampled_message_mae => nn_vs_sampled_message_mae,
		:nn_vs_sampled_marginal_mape => nn_vs_sampled_marginal_mape,
		:nn_vs_sampled_message_mape => nn_vs_sampled_message_mape,
		:sampled_vs_exact_marginal_rmse => sampled_vs_exact_marginal_rmse,
		:sampled_vs_exact_message_rmse => sampled_vs_exact_message_rmse,
		:sampled_vs_exact_marginal_mae => sampled_vs_exact_marginal_mae,
		:sampled_vs_exact_message_mae => sampled_vs_exact_message_mae,
		:sampled_vs_exact_marginal_mape => sampled_vs_exact_marginal_mape,
		:sampled_vs_exact_message_mape => sampled_vs_exact_message_mape,
	)

end

function write_results_to_file(metric_tuples, commit_hash, factor, target, gridpoints, gridpoints_factor)
	for (metrics, variable) in metric_tuples
		for i in 1:size(metrics[:nn_vs_exact_marginal_rmse], 1)
			if factor == "wsf"
				if variable == "X" || variable == "Y" || variable == "Z"
					current_gridpoint = gridpoints[:, i]
				else
					current_gridpoint = gridpoints_factor[:, i]
				end
			elseif factor == "gmf"
				if variable == "X" || variable == "Y"
					current_gridpoint = gridpoints[:, i]
				else
					current_gridpoint = gridpoints_factor[i]
				end
			end
			df = DataFrame(
				"Commit_Hash" => commit_hash,
				"Source" => "Prediction_vs_Exact",
				"Fixed" => target,
				"Variable" => variable,
				"Factor" => factor,
				"Gridpoint" => "$current_gridpoint",
				"RMSE_Marginal_Mean_mean" => metrics[:nn_vs_exact_marginal_rmse][i][1][1],
				"RMSE_Marginal_Mean_std" => metrics[:nn_vs_exact_marginal_rmse][i][2][1],
				"RMSE_Marginal_Variance_mean" => metrics[:nn_vs_exact_marginal_rmse][i][1][2],
				"RMSE_Marginal_Variance_std" => metrics[:nn_vs_exact_marginal_rmse][i][2][2],
				"RMSE_Marginal_Rho_mean" => metrics[:nn_vs_exact_marginal_rmse][i][1][3],
				"RMSE_Marginal_Rho_std" => metrics[:nn_vs_exact_marginal_rmse][i][2][3],
				"RMSE_Marginal_Tau_mean" => metrics[:nn_vs_exact_marginal_rmse][i][1][4],
				"RMSE_Marginal_Tau_std" => metrics[:nn_vs_exact_marginal_rmse][i][2][4],
				"RMSE_Msg_Mean_mean" => metrics[:nn_vs_exact_message_rmse][i][1][1],
				"RMSE_Msg_Mean_std" => metrics[:nn_vs_exact_message_rmse][i][2][1],
				"RMSE_Msg_Variance_mean" => metrics[:nn_vs_exact_message_rmse][i][1][2],
				"RMSE_Msg_Variance_std" => metrics[:nn_vs_exact_message_rmse][i][2][2],
				"RMSE_Msg_Rho_mean" => metrics[:nn_vs_exact_message_rmse][i][1][3],
				"RMSE_Msg_Rho_std" => metrics[:nn_vs_exact_message_rmse][i][2][3],
				"RMSE_Msg_Tau_mean" => metrics[:nn_vs_exact_message_rmse][i][1][4],
				"RMSE_Msg_Tau_std" => metrics[:nn_vs_exact_message_rmse][i][2][4],
				"MAE_Marginal_Mean_mean" => metrics[:nn_vs_exact_marginal_mae][i][1][1],
				"MAE_Marginal_Mean_std" => metrics[:nn_vs_exact_marginal_mae][i][2][1],
				"MAE_Marginal_Variance_mean" => metrics[:nn_vs_exact_marginal_mae][i][1][2],
				"MAE_Marginal_Variance_std" => metrics[:nn_vs_exact_marginal_mae][i][2][2],
				"MAE_Marginal_Rho_mean" => metrics[:nn_vs_exact_marginal_mae][i][1][3],
				"MAE_Marginal_Rho_std" => metrics[:nn_vs_exact_marginal_mae][i][2][3],
				"MAE_Marginal_Tau_mean" => metrics[:nn_vs_exact_marginal_mae][i][1][4],
				"MAE_Marginal_Tau_std" => metrics[:nn_vs_exact_marginal_mae][i][2][4],
				"MAE_Msg_Mean_mean" => metrics[:nn_vs_exact_message_mae][i][1][1],
				"MAE_Msg_Mean_std" => metrics[:nn_vs_exact_message_mae][i][2][1],
				"MAE_Msg_Variance_mean" => metrics[:nn_vs_exact_message_mae][i][1][2],
				"MAE_Msg_Variance_std" => metrics[:nn_vs_exact_message_mae][i][2][2],
				"MAE_Msg_Rho_mean" => metrics[:nn_vs_exact_message_mae][i][1][3],
				"MAE_Msg_Rho_std" => metrics[:nn_vs_exact_message_mae][i][2][3],
				"MAE_Msg_Tau_mean" => metrics[:nn_vs_exact_message_mae][i][1][4],
				"MAE_Msg_Tau_std" => metrics[:nn_vs_exact_message_mae][i][2][4],
				"MAPE_Marginal_Mean_mean" => metrics[:nn_vs_exact_marginal_mape][i][1][1],
				"MAPE_Marginal_Mean_std" => metrics[:nn_vs_exact_marginal_mape][i][2][1],
				"MAPE_Marginal_Variance_mean" => metrics[:nn_vs_exact_marginal_mape][i][1][2],
				"MAPE_Marginal_Variance_std" => metrics[:nn_vs_exact_marginal_mape][i][2][2],
				"MAPE_Marginal_Rho_mean" => metrics[:nn_vs_exact_marginal_mape][i][1][3],
				"MAPE_Marginal_Rho_std" => metrics[:nn_vs_exact_marginal_mape][i][2][3],
				"MAPE_Marginal_Tau_mean" => metrics[:nn_vs_exact_marginal_mape][i][1][4],
				"MAPE_Marginal_Tau_std" => metrics[:nn_vs_exact_marginal_mape][i][2][4],
				"MAPE_Msg_Mean_mean" => metrics[:nn_vs_exact_message_mape][i][1][1],
				"MAPE_Msg_Mean_std" => metrics[:nn_vs_exact_message_mape][i][2][1],
				"MAPE_Msg_Variance_mean" => metrics[:nn_vs_exact_message_mape][i][1][2],
				"MAPE_Msg_Variance_std" => metrics[:nn_vs_exact_message_mape][i][2][2],
				"MAPE_Msg_Rho_mean" => metrics[:nn_vs_exact_message_mape][i][1][3],
				"MAPE_Msg_Rho_std" => metrics[:nn_vs_exact_message_mape][i][2][3],
				"MAPE_Msg_Tau_mean" => metrics[:nn_vs_exact_message_mape][i][1][4],
				"MAPE_Msg_Tau_std" => metrics[:nn_vs_exact_message_mape][i][2][4],
			)
			if isfile(filepath)
				CSV.write(filepath, df; append=true)
			else
				CSV.write(filepath, df)
			end
			df = DataFrame(
				"Commit_Hash" => commit_hash,
				"Source" => "Sampled_vs_Exact",
				"Fixed" => target,
				"Variable" => variable,
				"Factor" => factor,
				"Gridpoint" => "$current_gridpoint",
				"RMSE_Marginal_Mean_mean" => metrics[:sampled_vs_exact_marginal_rmse][i][1][1],
				"RMSE_Marginal_Mean_std" => metrics[:sampled_vs_exact_marginal_rmse][i][2][1],
				"RMSE_Marginal_Variance_mean" => metrics[:sampled_vs_exact_marginal_rmse][i][1][2],
				"RMSE_Marginal_Variance_std" => metrics[:sampled_vs_exact_marginal_rmse][i][2][2],
				"RMSE_Marginal_Rho_mean" => metrics[:sampled_vs_exact_marginal_rmse][i][1][3],
				"RMSE_Marginal_Rho_std" => metrics[:sampled_vs_exact_marginal_rmse][i][2][3],
				"RMSE_Marginal_Tau_mean" => metrics[:sampled_vs_exact_marginal_rmse][i][1][4],
				"RMSE_Marginal_Tau_std" => metrics[:sampled_vs_exact_marginal_rmse][i][2][4],
				"RMSE_Msg_Mean_mean" => metrics[:sampled_vs_exact_message_rmse][i][1][1],
				"RMSE_Msg_Mean_std" => metrics[:sampled_vs_exact_message_rmse][i][2][1],
				"RMSE_Msg_Variance_mean" => metrics[:sampled_vs_exact_message_rmse][i][1][2],
				"RMSE_Msg_Variance_std" => metrics[:sampled_vs_exact_message_rmse][i][2][2],
				"RMSE_Msg_Rho_mean" => metrics[:sampled_vs_exact_message_rmse][i][1][3],
				"RMSE_Msg_Rho_std" => metrics[:sampled_vs_exact_message_rmse][i][2][3],
				"RMSE_Msg_Tau_mean" => metrics[:sampled_vs_exact_message_rmse][i][1][4],
				"RMSE_Msg_Tau_std" => metrics[:sampled_vs_exact_message_rmse][i][2][4],
				"MAE_Marginal_Mean_mean" => metrics[:sampled_vs_exact_marginal_mae][i][1][1],
				"MAE_Marginal_Mean_std" => metrics[:sampled_vs_exact_marginal_mae][i][2][1],
				"MAE_Marginal_Variance_mean" => metrics[:sampled_vs_exact_marginal_mae][i][1][2],
				"MAE_Marginal_Variance_std" => metrics[:sampled_vs_exact_marginal_mae][i][2][2],
				"MAE_Marginal_Rho_mean" => metrics[:sampled_vs_exact_marginal_mae][i][1][3],
				"MAE_Marginal_Rho_std" => metrics[:sampled_vs_exact_marginal_mae][i][2][3],
				"MAE_Marginal_Tau_mean" => metrics[:sampled_vs_exact_marginal_mae][i][1][4],
				"MAE_Marginal_Tau_std" => metrics[:sampled_vs_exact_marginal_mae][i][2][4],
				"MAE_Msg_Mean_mean" => metrics[:sampled_vs_exact_message_mae][i][1][1],
				"MAE_Msg_Mean_std" => metrics[:sampled_vs_exact_message_mae][i][2][1],
				"MAE_Msg_Variance_mean" => metrics[:sampled_vs_exact_message_mae][i][1][2],
				"MAE_Msg_Variance_std" => metrics[:sampled_vs_exact_message_mae][i][2][2],
				"MAE_Msg_Rho_mean" => metrics[:sampled_vs_exact_message_mae][i][1][3],
				"MAE_Msg_Rho_std" => metrics[:sampled_vs_exact_message_mae][i][2][3],
				"MAE_Msg_Tau_mean" => metrics[:sampled_vs_exact_message_mae][i][1][4],
				"MAE_Msg_Tau_std" => metrics[:sampled_vs_exact_message_mae][i][2][4],
				"MAPE_Marginal_Mean_mean" => metrics[:sampled_vs_exact_marginal_mape][i][1][1],
				"MAPE_Marginal_Mean_std" => metrics[:sampled_vs_exact_marginal_mape][i][2][1],
				"MAPE_Marginal_Variance_mean" => metrics[:sampled_vs_exact_marginal_mape][i][1][2],
				"MAPE_Marginal_Variance_std" => metrics[:sampled_vs_exact_marginal_mape][i][2][2],
				"MAPE_Marginal_Rho_mean" => metrics[:sampled_vs_exact_marginal_mape][i][1][3],
				"MAPE_Marginal_Rho_std" => metrics[:sampled_vs_exact_marginal_mape][i][2][3],
				"MAPE_Marginal_Tau_mean" => metrics[:sampled_vs_exact_marginal_mape][i][1][4],
				"MAPE_Marginal_Tau_std" => metrics[:sampled_vs_exact_marginal_mape][i][2][4],
				"MAPE_Msg_Mean_mean" => metrics[:sampled_vs_exact_message_mape][i][1][1],
				"MAPE_Msg_Mean_std" => metrics[:sampled_vs_exact_message_mape][i][2][1],
				"MAPE_Msg_Variance_mean" => metrics[:sampled_vs_exact_message_mape][i][1][2],
				"MAPE_Msg_Variance_std" => metrics[:sampled_vs_exact_message_mape][i][2][2],
				"MAPE_Msg_Rho_mean" => metrics[:sampled_vs_exact_message_mape][i][1][3],
				"MAPE_Msg_Rho_std" => metrics[:sampled_vs_exact_message_mape][i][2][3],
				"MAPE_Msg_Tau_mean" => metrics[:sampled_vs_exact_message_mape][i][1][4],
				"MAPE_Msg_Tau_std" => metrics[:sampled_vs_exact_message_mape][i][2][4],
			)
			if isfile(filepath)
				CSV.write(filepath, df; append=true)
			else
				CSV.write(filepath, df)
			end
		end
	end
end

for factor in factors
	if factor == "wsf"
		standard_input = [1, 2, 1, 2, 0, 2, 1, -1, 0]
		for target in ["targets_X", "targets_Y", "targets_Z"]
			if target == "targets_X"
				# create a Matrix variable y by copy&paste standard_input along the rows
				variable_Y = Matrix{Float64}(repeat(standard_input, 1, n_points))
				variable_Z = Matrix{Float64}(repeat(standard_input, 1, n_points))
				variable_a_b = Matrix{Float64}(repeat(standard_input, 1, n_points_constants))

				# replace the mean and var of the target variable with the gridpoints
				variable_Y[3:4, :] = gridpoints
				variable_Z[5:6, :] = gridpoints
				variable_a_b[7:8, :] = gridpoints_constants

				# get output samples for inputs
				t = time()
				analytical_marginals_variable_Y, analytical_messages_variable_Y = get_analytical_output(variable_Y, WeightedSumFactorGeneration.X, WeightedSumFactorGeneration.calc_msg_x, gridpoints)
				analytical_marginals_variable_Z, analytical_messages_variable_Z = get_analytical_output(variable_Z, WeightedSumFactorGeneration.X, WeightedSumFactorGeneration.calc_msg_x, gridpoints)
				analytical_marginals_variable_a_b, analytical_messages_variable_a_b = get_analytical_output(variable_a_b, WeightedSumFactorGeneration.X, WeightedSumFactorGeneration.calc_msg_x, gridpoints_constants)
				println("Elapsed time for analytical $(factor) $(target): ", time() - t)


				t = time()
				sampled_marginals_variable_Y, sampled_messages_variable_Y = get_sampled_output_wsf(variable_Y, WeightedSumFactorGeneration.X, factor, gridpoints)
				sampled_marginals_variable_Z, sampled_messages_variable_Z = get_sampled_output_wsf(variable_Z, WeightedSumFactorGeneration.X, factor, gridpoints)
				sampled_marginals_variable_a_b, sampled_messages_variable_a_b = get_sampled_output_wsf(variable_a_b, WeightedSumFactorGeneration.X, factor, gridpoints_constants)
				println("Elapsed time for sampling $(factor) $(target): ", time() - t)

				t = time()
				models = nn_map[factor][target]
				nn_output_variable_Y_marginals, nn_output_variable_Y_msgs = get_predicted_output_wsf(variable_Y, models, WeightedSumFactorGeneration.X, gridpoints)
				nn_output_variable_Z_marginals, nn_output_variable_Z_msgs = get_predicted_output_wsf(variable_Z, models, WeightedSumFactorGeneration.X, gridpoints)
				nn_output_variable_a_b_marginals, nn_output_variable_a_b_msgs = get_predicted_output_wsf(variable_a_b, models, WeightedSumFactorGeneration.X, gridpoints_constants)
				println("Elapsed time for nn $(factor) $(target): ", time() - t)

				t = time()
				metrics_variable_Y = calc_metrics((nn_output_variable_Y_marginals, nn_output_variable_Y_msgs), (sampled_marginals_variable_Y, sampled_messages_variable_Y), (analytical_marginals_variable_Y, analytical_messages_variable_Y), gridpoints)
				metrics_variable_Z = calc_metrics((nn_output_variable_Z_marginals, nn_output_variable_Z_msgs), (sampled_marginals_variable_Z, sampled_messages_variable_Z), (analytical_marginals_variable_Z, analytical_messages_variable_Z), gridpoints)
				metrics_variable_a_b =
					calc_metrics((nn_output_variable_a_b_marginals, nn_output_variable_a_b_msgs), (sampled_marginals_variable_a_b, sampled_messages_variable_a_b), (analytical_marginals_variable_a_b, analytical_messages_variable_a_b), gridpoints_constants)
				println("Elapsed time for metrics $(factor) $(target): ", time() - t)

				# save metrics
				write_results_to_file([(metrics_variable_Y, "Y"), (metrics_variable_Z, "Z"), (metrics_variable_a_b, "a_b")], commit_hash, factor, target, gridpoints, gridpoints_constants)

			elseif target == "targets_Y"
				# create a Matrix variable y by copy&paste standard_input along the rows
				variable_X = Matrix{Float64}(repeat(standard_input, 1, n_points))
				variable_Z = Matrix{Float64}(repeat(standard_input, 1, n_points))
				variable_a_b = Matrix{Float64}(repeat(standard_input, 1, n_points_constants))

				# replace the mean and var of the target variable with the gridpoints
				variable_X[1:2, :] = gridpoints
				variable_Z[5:6, :] = gridpoints
				variable_a_b[7:8, :] = gridpoints_constants

				# get output samples for inputs
				t = time()
				analytical_marginals_variable_X, analytical_messages_variable_X = get_analytical_output(variable_X, WeightedSumFactorGeneration.Y, WeightedSumFactorGeneration.calc_msg_y, gridpoints)
				analytical_marginals_variable_Z, analytical_messages_variable_Z = get_analytical_output(variable_Z, WeightedSumFactorGeneration.Y, WeightedSumFactorGeneration.calc_msg_y, gridpoints)
				analytical_marginals_variable_a_b, analytical_messages_variable_a_b = get_analytical_output(variable_a_b, WeightedSumFactorGeneration.Y, WeightedSumFactorGeneration.calc_msg_y, gridpoints_constants)
				println("Elapsed time for analytical $(factor) $(target): ", time() - t)


				t = time()
				sampled_marginals_variable_X, sampled_messages_variable_X = get_sampled_output_wsf(variable_X, WeightedSumFactorGeneration.Y, factor, gridpoints)
				sampled_marginals_variable_Z, sampled_messages_variable_Z = get_sampled_output_wsf(variable_Z, WeightedSumFactorGeneration.Y, factor, gridpoints)
				sampled_marginals_variable_a_b, sampled_messages_variable_a_b = get_sampled_output_wsf(variable_a_b, WeightedSumFactorGeneration.Y, factor, gridpoints_constants)
				println("Elapsed time for sampling $(factor) $(target): ", time() - t)

				t = time()
				models = nn_map[factor][target]
				nn_output_variable_X_marginals, nn_output_variable_X_msgs = get_predicted_output_wsf(variable_X, models, WeightedSumFactorGeneration.Y, gridpoints)
				nn_output_variable_Z_marginals, nn_output_variable_Z_msgs = get_predicted_output_wsf(variable_Z, models, WeightedSumFactorGeneration.Y, gridpoints)
				nn_output_variable_a_b_marginals, nn_output_variable_a_b_msgs = get_predicted_output_wsf(variable_a_b, models, WeightedSumFactorGeneration.Y, gridpoints_constants)
				println("Elapsed time for nn $(factor) $(target): ", time() - t)

				t = time()
				metrics_variable_X = calc_metrics((nn_output_variable_X_marginals, nn_output_variable_X_msgs), (sampled_marginals_variable_X, sampled_messages_variable_X), (analytical_marginals_variable_X, analytical_messages_variable_X), gridpoints)
				metrics_variable_Z = calc_metrics((nn_output_variable_Z_marginals, nn_output_variable_Z_msgs), (sampled_marginals_variable_Z, sampled_messages_variable_Z), (analytical_marginals_variable_Z, analytical_messages_variable_Z), gridpoints)
				metrics_variable_a_b =
					calc_metrics((nn_output_variable_a_b_marginals, nn_output_variable_a_b_msgs), (sampled_marginals_variable_a_b, sampled_messages_variable_a_b), (analytical_marginals_variable_a_b, analytical_messages_variable_a_b), gridpoints_constants)
				println("Elapsed time for metrics $(factor) $(target): ", time() - t)

				# save metrics
				write_results_to_file([(metrics_variable_X, "X"), (metrics_variable_Z, "Z"), (metrics_variable_a_b, "a_b")], commit_hash, factor, target, gridpoints, gridpoints_constants)

			elseif target == "targets_Z"
				# create a Matrix variable y by copy&paste standard_input along the rows
				variable_X = Matrix{Float64}(repeat(standard_input, 1, n_points))
				variable_Y = Matrix{Float64}(repeat(standard_input, 1, n_points))
				variable_a_b = Matrix{Float64}(repeat(standard_input, 1, n_points_constants))

				# replace the mean and var of the target variable with the gridpoints
				variable_X[1:2, :] = gridpoints
				variable_Y[3:4, :] = gridpoints
				variable_a_b[7:8, :] = gridpoints_constants

				# get output samples for inputs
				t = time()
				analytical_marginals_variable_X, analytical_messages_variable_X = get_analytical_output(variable_X, WeightedSumFactorGeneration.Z, WeightedSumFactorGeneration.calc_msg_z, gridpoints)
				analytical_marginals_variable_Y, analytical_messages_variable_Y = get_analytical_output(variable_Y, WeightedSumFactorGeneration.Z, WeightedSumFactorGeneration.calc_msg_z, gridpoints)
				analytical_marginals_variable_a_b, analytical_messages_variable_a_b = get_analytical_output(variable_a_b, WeightedSumFactorGeneration.Z, WeightedSumFactorGeneration.calc_msg_z, gridpoints_constants)
				println("Elapsed time for analytical $(factor) $(target): ", time() - t)


				t = time()
				sampled_marginals_variable_X, sampled_messages_variable_X = get_sampled_output_wsf(variable_X, WeightedSumFactorGeneration.Z, factor, gridpoints)
				sampled_marginals_variable_Y, sampled_messages_variable_Y = get_sampled_output_wsf(variable_Y, WeightedSumFactorGeneration.Z, factor, gridpoints)
				sampled_marginals_variable_a_b, sampled_messages_variable_a_b = get_sampled_output_wsf(variable_a_b, WeightedSumFactorGeneration.Z, factor, gridpoints_constants)
				println("Elapsed time for sampling $(factor) $(target): ", time() - t)

				t = time()
				models = nn_map[factor][target]
				nn_output_variable_X_marginals, nn_output_variable_X_msgs = get_predicted_output_wsf(variable_X, models, WeightedSumFactorGeneration.Z, gridpoints)
				nn_output_variable_Y_marginals, nn_output_variable_Y_msgs = get_predicted_output_wsf(variable_Y, models, WeightedSumFactorGeneration.Z, gridpoints)
				nn_output_variable_a_b_marginals, nn_output_variable_a_b_msgs = get_predicted_output_wsf(variable_a_b, models, WeightedSumFactorGeneration.Z, gridpoints_constants)
				println("Elapsed time for nn $(factor) $(target): ", time() - t)

				t = time()
				metrics_variable_X = calc_metrics((nn_output_variable_X_marginals, nn_output_variable_X_msgs), (sampled_marginals_variable_X, sampled_messages_variable_X), (analytical_marginals_variable_X, analytical_messages_variable_X), gridpoints)
				metrics_variable_Y = calc_metrics((nn_output_variable_Y_marginals, nn_output_variable_Y_msgs), (sampled_marginals_variable_Y, sampled_messages_variable_Y), (analytical_marginals_variable_Y, analytical_messages_variable_Y), gridpoints)
				metrics_variable_a_b =
					calc_metrics((nn_output_variable_a_b_marginals, nn_output_variable_a_b_msgs), (sampled_marginals_variable_a_b, sampled_messages_variable_a_b), (analytical_marginals_variable_a_b, analytical_messages_variable_a_b), gridpoints_constants)
				println("Elapsed time for metrics $(factor) $(target): ", time() - t)

				# save metrics
				write_results_to_file([(metrics_variable_X, "X"), (metrics_variable_Y, "Y"), (metrics_variable_a_b, "a_b")], commit_hash, factor, target, gridpoints, gridpoints_constants)
			else
				error("Unknown target: $target")
			end
		end

	elseif factor == "gmf"
		standard_input = [1, 2, 1, 2, 2]
		for target in ["targets_X", "targets_Y"]
			if target == "targets_X"
				# create a Matrix variable y by copy&paste standard_input along the rows
				variable_Y = Matrix{Float64}(repeat(standard_input, 1, n_points))
				variable_beta = Matrix{Float64}(repeat(standard_input, 1, n_points_beta))

				# replace the mean and var of the target variable with the gridpoints
				variable_Y[3:4, :] = gridpoints
				variable_beta[5, :] = gridpoints_beta

				# get output samples for inputs
				t = time()
				analytical_marginals_variable_Y, analytical_messages_variable_Y = get_analytical_output(variable_Y, GaussianMeanFactorGeneration.X, GaussianMeanFactorGeneration.calc_msg_x, gridpoints)
				analytical_marginals_variable_beta, analytical_messages_variable_beta = get_analytical_output(variable_beta, GaussianMeanFactorGeneration.X, GaussianMeanFactorGeneration.calc_msg_x, gridpoints_beta)
				println("Elapsed time for analytical $(factor) $(target): ", time() - t)


				t = time()
				sampled_marginals_variable_Y, sampled_messages_variable_Y = get_sampled_output_gmf(variable_Y, GaussianMeanFactorGeneration.X, factor, gridpoints)
				sampled_marginals_variable_beta, sampled_messages_variable_beta = get_sampled_output_gmf(variable_beta, GaussianMeanFactorGeneration.X, factor, gridpoints_beta)
				println("Elapsed time for sampling $(factor) $(target): ", time() - t)

				t = time()
				models = nn_map[factor][target]
				nn_output_variable_Y_marginals, nn_output_variable_Y_msgs = get_predicted_output_gmf(variable_Y, models, GaussianMeanFactorGeneration.X, gridpoints)
				nn_output_variable_beta_marginals, nn_output_variable_beta_msgs = get_predicted_output_gmf(variable_beta, models, GaussianMeanFactorGeneration.X, gridpoints_beta)
				println("Elapsed time for nn $(factor) $(target): ", time() - t)

				t = time()
				metrics_variable_Y = calc_metrics((nn_output_variable_Y_marginals, nn_output_variable_Y_msgs), (sampled_marginals_variable_Y, sampled_messages_variable_Y), (analytical_marginals_variable_Y, analytical_messages_variable_Y), gridpoints)
				metrics_variable_beta = calc_metrics((nn_output_variable_beta_marginals, nn_output_variable_beta_msgs), (sampled_marginals_variable_beta, sampled_messages_variable_beta), (analytical_marginals_variable_beta, analytical_messages_variable_beta), gridpoints_beta)
				println("Elapsed time for metrics $(factor) $(target): ", time() - t)

				write_results_to_file([(metrics_variable_Y, "Y"), (metrics_variable_beta, "beta^2")], commit_hash, factor, target, gridpoints, gridpoints_beta)
				
				
			elseif target == "targets_Y"
				# create a Matrix variable y by copy&paste standard_input along the rows
				variable_X = Matrix{Float64}(repeat(standard_input, 1, n_points))
				variable_beta = Matrix{Float64}(repeat(standard_input, 1, n_points_beta))

				# replace the mean and var of the target variable with the gridpoints
				variable_X[1:2, :] = gridpoints
				variable_beta[5, :] = gridpoints_beta

				# get output samples for inputs
				t = time()
				analytical_marginals_variable_X, analytical_messages_variable_X = get_analytical_output(variable_X, GaussianMeanFactorGeneration.Y, GaussianMeanFactorGeneration.calc_msg_y, gridpoints)
				analytical_marginals_variable_beta, analytical_messages_variable_beta = get_analytical_output(variable_beta, GaussianMeanFactorGeneration.Y, GaussianMeanFactorGeneration.calc_msg_y, gridpoints_beta)
				println("Elapsed time for analytical $(factor) $(target): ", time() - t)


				t = time()
				sampled_marginals_variable_X, sampled_messages_variable_X = get_sampled_output_gmf(variable_X, GaussianMeanFactorGeneration.Y, factor, gridpoints)
				sampled_marginals_variable_beta, sampled_messages_variable_beta = get_sampled_output_gmf(variable_beta, GaussianMeanFactorGeneration.Y, factor, gridpoints_beta)
				println("Elapsed time for sampling $(factor) $(target): ", time() - t)

				t = time()
				models = nn_map[factor][target]
				nn_output_variable_X_marginals, nn_output_variable_X_msgs = get_predicted_output_gmf(variable_X, models, GaussianMeanFactorGeneration.Y, gridpoints)
				nn_output_variable_beta_marginals, nn_output_variable_beta_msgs = get_predicted_output_gmf(variable_beta, models, GaussianMeanFactorGeneration.X, gridpoints_beta)
				println("Elapsed time for nn $(factor) $(target): ", time() - t)

				t = time()
				metrics_variable_X = calc_metrics((nn_output_variable_X_marginals, nn_output_variable_X_msgs), (sampled_marginals_variable_X, sampled_messages_variable_X), (analytical_marginals_variable_X, analytical_messages_variable_X), gridpoints)
				metrics_variable_beta =
					calc_metrics((nn_output_variable_beta_marginals, nn_output_variable_beta_msgs), (sampled_marginals_variable_beta, sampled_messages_variable_beta), (analytical_marginals_variable_beta, analytical_messages_variable_beta), gridpoints_beta)
				println("Elapsed time for metrics $(factor) $(target): ", time() - t)

				# save metrics
				write_results_to_file([(metrics_variable_X, "X"), (metrics_variable_beta, "beta^2")], commit_hash, factor, target, gridpoints, gridpoints_beta)
			end
		end
	end
end
