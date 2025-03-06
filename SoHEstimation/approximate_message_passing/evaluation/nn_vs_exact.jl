include("../nn/mjl.jl")
using .NN: run_experiment, get_data, remove_outliers, rmse_loss, rmspe_loss
using ProgressMeter
using Random
using Format
using CSV
using DataFrames

seed = 707542
Random.seed!(seed)
nfolds = 1
max_epochs = 2_000

data_map = Dict{String, Any}()

ns = [20_000, 10_000, 5_000, 2_000, 1_000, 500, 200, 100]
factors = [:wsf, :gmf]

combinations = [(n, factor) for n in ns, factor in factors]
n_combinations = length(combinations)

progressbar = Progress(n_combinations, desc = "Loading data...")
for (n, factor) in combinations
	if factor == :wsf
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
			samples_per_inputs = [1_000_000, 10_000_000, 1_000_000, 100_000, 10_000, 1_000]
		else
			samples_per_inputs = [1_000_000]
		end
		samples_per_inputs = [1_000_000]

		for samples_per_input in samples_per_inputs
			nstring = replace(format(n, commas = true), "," => "_")
			for i in 1:3
				filename = factor_path * "$(nstring)_$(samples_per_input)_Experiment_$i.jld2"
				datapath = "SoHEstimation/approximate_message_passing/data_masterarbeit/" * filename

				# Check if the file exists
				if !isfile(datapath)
					error("File not found: $datapath")
				end
				X, y = get_data(datapath; target = target, factor = factor, transform_to_tau_rho = false, verbose = 0)
				X, y, _, _ = remove_outliers(X, y; verbose = 0)

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

function update_csv(df::DataFrame, filepath::String)
	# Check if file exists
	if isfile(filepath)
		existing_df = CSV.read(filepath, DataFrame)

		# Identify matching rows
		mask =
			(existing_df.Source .== df.Source[1]) .&&
			(existing_df.Factor .== df.Factor[1]) .&&
			(existing_df.Target .== df.Target[1]) .&&
			(existing_df.N .== df.N[1]) .&&
			(existing_df.SamplesPerInput .== df.SamplesPerInput[1]) .&&
			(existing_df.Experiment .== df.Experiment[1])

		if any(mask)
			# Update the matching row in-place
			existing_df[mask, :] .= df
		else
			# Append the new row if no match is found
			existing_df = vcat(existing_df, df)
		end

		# Write updated DataFrame back to CSV
		CSV.write(filepath, existing_df)
	else
		# If file does not exist, create it
		CSV.write(filepath, df)
	end
end

function config_exists(source, factor, n, samples_per_input, target, i; filepath="")
	if isfile(filepath)
		df = CSV.read(filepath, DataFrame)
	else
		df = DataFrame()
	end

    if isempty(df)
        return false
    end
    mask =
        (df.Factor .== factor) .&&
		(df.Source .== source) .&&
        (df.N .== n) .&&
        (df.SamplesPerInput .== samples_per_input) .&&
        (df.Target .== target) .&&
        (df.Experiment .== i)

    return any(mask)
end


progressbar = Progress(length(data_map) * nfolds, desc = "Running experiments for $(length(data_map)) datasets on $nfolds folds ...")


params_dict = Dict(
	"gmf_X" => Dict(
		:n_neurons => 2024,
		:n_layers => 2,
		:activation_function => "relu",
		:batch_size => 512,
		:output_layer_choice => "relu",
		:target => "targets_X",
		:moment => :both,
		:transform_to_tau_rho => false,
		:scale => :minmax,
		:scale_output => :minmax,
		:loss_function => rmse_loss,
		:factor => "gmf",
	),
	"gmf_Y" => Dict(
		:n_neurons => 2024,
		:n_layers => 3,
		:activation_function => "relu",
		:batch_size => 512,
		:output_layer_choice => "softplus",
		:target => "targets_Y",
		:moment => :both,
		:transform_to_tau_rho => false,
		:scale => :minmax,
		:scale_output => :minmax,
		:loss_function => rmse_loss,
		:factor => "gmf",
	),
	"wsf_X" => Dict(
		:n_neurons => 512,
		:n_layers => 2,
		:activation_function => "tanh_fast",
		:batch_size => 512,
		:output_layer_choice => "relu",
		:target => "targets_X",
		:moment => :both,
		:transform_to_tau_rho => false,
		:scale => :zscore,
		:scale_output => :zscore,
		:loss_function => rmse_loss,
		:factor => "wsf",
	),
	"wsf_Y" => Dict(
		:n_neurons => 512,
		:n_layers => 2,
		:activation_function => "tanh_fast",
		:batch_size => 512,
		:output_layer_choice => "softplus",
		:target => "targets_Y",
		:moment => :both,
		:transform_to_tau_rho => false,
		:scale => :zscore,
		:scale_output => :zscore,
		:loss_function => rmse_loss,
		:factor => "wsf",
	),
	"wsf_Z" => Dict(
		:n_neurons => 512,
		:n_layers => 1,
		:activation_function => "tanh_fast",
		:batch_size => 512,
		:output_layer_choice => "softplus",
		:target => "targets_Z",
		:moment => :both,
		:transform_to_tau_rho => false,
		:scale => :zscore,
		:scale_output => :minmax,
		:loss_function => rmse_loss,
		:factor => "wsf",
	),
)

# Define the output file path
filepath = "SoHEstimation/approximate_message_passing/evaluation/results/metrics_marginal_vs_exact_MA.csv"
for (key, (X, y)) in data_map
	factor, n, samples_per_input, target, variable, i = split(key, "_")
	target = target * "_$(variable)"

	params = deepcopy(params_dict[factor*"_$(variable)"])
	params[:key] = key

	println("Key: $key")

	if config_exists("Prediction_vs_Exact", factor, n, samples_per_input, target, i, filepath=filepath)
        println("Skipping existing configuration: $key")
        continue
    end

	metrics = run_experiment(X, y, params, nfolds, seed, max_epochs, :both, target; progressbar = progressbar)

	continue
	df = DataFrame(
		"Commit_Hash" => commit_hash,
		"Source" => "Prediction_vs_Exact",
		"Factor" => factor,
		"N" => n,
		"SamplesPerInput" => samples_per_input,
		"Experiment" => i,
		"Target" => target,
		"RMSE_Marginal_Mean" => metrics[:rmse_marginal_prediction_exact_mean],
		"RMSE_Marginal_Variance" => metrics[:rmse_marginal_prediction_exact_variance],
		"RMSE_Marginal_Rho" => metrics[:rmse_marginal_prediction_exact_rho],
		"RMSE_Marginal_Tau" => metrics[:rmse_marginal_prediction_exact_tau],
		"RMSE_Msg_Mean" => metrics[:rmse_message_prediction_exact_mean],
		"RMSE_Msg_Variance" => metrics[:rmse_message_prediction_exact_variance],
		"RMSE_Msg_Rho" => metrics[:rmse_message_prediction_exact_rho],
		"RMSE_Msg_Tau" => metrics[:rmse_message_prediction_exact_tau],
		"MAE_Marginal_Mean" => metrics[:mae_marginal_prediction_exact_mean],
		"MAE_Marginal_Variance" => metrics[:mae_marginal_prediction_exact_variance],
		"MAE_Marginal_Rho" => metrics[:mae_marginal_prediction_exact_rho],
		"MAE_Marginal_Tau" => metrics[:mae_marginal_prediction_exact_tau],
		"MAE_Msg_Mean" => metrics[:mae_message_prediction_exact_mean],
		"MAE_Msg_Variance" => metrics[:mae_message_prediction_exact_variance],
		"MAE_Msg_Rho" => metrics[:mae_message_prediction_exact_rho],
		"MAE_Msg_Tau" => metrics[:mae_message_prediction_exact_tau],
		"MAPE_Marginal_Mean" => metrics[:mape_marginal_prediction_exact_mean],
		"MAPE_Marginal_Variance" => metrics[:mape_marginal_prediction_exact_variance],
		"MAPE_Marginal_Rho" => metrics[:mape_marginal_prediction_exact_rho],
		"MAPE_Marginal_Tau" => metrics[:mape_marginal_prediction_exact_tau],
		"MAPE_Msg_Mean" => metrics[:mape_message_prediction_exact_mean],
		"MAPE_Msg_Variance" => metrics[:mape_message_prediction_exact_variance],
		"MAPE_Msg_Rho" => metrics[:mape_message_prediction_exact_rho],
		"MAPE_Msg_Tau" => metrics[:mape_message_prediction_exact_tau],
	)

	# Update the CSV file with the new data
	update_csv(df, filepath)
end
